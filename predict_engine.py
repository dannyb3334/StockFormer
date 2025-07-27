
import torch
import yaml
from StockFormer import create_compiled_stockformer, output_to_signals
from preprocess import create_predict_data

class StockFormerPredictor:
    def __init__(self, model_path, config_path, tickers, lag, lead, standard_window, lookahead=0):
        """
        Initialize the predictor with model/config paths and prediction parameters.
        Loads model parameters from config and initializes the model.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.config_path = config_path
        self.tickers = tickers
        self.lag = lag
        self.lead = lead
        self.lookahead = lookahead
        self.standard_window = standard_window
        self.model = None
        self.model_params = None
        self._load_config()
        self._init_model()

    def _load_config(self):
        """
        Load model parameters from the YAML config file.
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.model_params = config.get('model_params', {})

    def _init_model(self):
        """
        Initialize the StockFormer model and load weights from checkpoint.
        """
        self.model = create_compiled_stockformer(self.device, **self.model_params)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self):
        """
        Run prediction for the latest data sample and print trading signals.
        Returns the predicted signals for the provided tickers.
        """
        X, T = create_predict_data(
            tickers=self.tickers,
            lag=self.lag,
            lead=self.lead,
            standard_window=self.standard_window
        )
        # Only use the last sample for prediction
        X_tensor = torch.FloatTensor(X[-1:]).to(self.device)
        T_tensor = torch.FloatTensor(T[-1:]).to(self.device)
        with torch.inference_mode():
            out = self.model(X_tensor, T_tensor)
        # Convert model output to trading signals
        signals = output_to_signals(out, lookahead=self.lookahead)
        actions = ["Hold", "Buy", "Sell"]
        for ticker, signal in zip(self.tickers, signals[0]):
            print(f"Ticker: {ticker}, Signal: {actions[int(signal)]}")
        return signals

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_path = config.get('model_path', 'stockformer_model.pth')
    model_params = config.get('model_params', {})
    tickers = model_params.get('tickers')
    lag = model_params.get('seq_len')
    lead = model_params.get('pred_len')
    standard_window = model_params.get('min_len_for_pred')
    lookahead = model_params.get('lookahead', 0)

    # Example usage: initialize predictor and run prediction
    predictor = StockFormerPredictor(
        model_path=model_path,
        config_path="config.yaml",
        tickers=tickers,
        lag=lag,
        lead=lead,
        standard_window=standard_window,
        lookahead=lookahead
    )
    predictor.predict()
