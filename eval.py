
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def metrics(predictions, labels):
    """
    Calculate evaluation metrics for the model predictions.
    Args:
        predictions (np.array): Array of predicted label pairs.
        labels (np.array): Array of true label pairs.

    Returns:
        dict: Dictionary containing accuracy and F1 score.
    """
    num_predictions = len(predictions)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='samples')

    IC = ((predictions['RETURN_RATE'] - np.mean(predictions['RETURN_RATE'])) * (labels['RETURN_RATE'] - np.mean(labels['RETURN_RATE']))) / (np.std(predictions['RETURN_RATE']) * np.std(labels['RETURN_RATE']) * len(predictions))
    ICIR = np.mean(IC) / np.std(IC)
    RANK_IC = np.corrcoef(predictions['RETURN_RATE'], labels['RETURN_RATE'])[0, 1]
    RANK_ICIR = np.mean(RANK_IC) / np.std(RANK_IC)

    directional_accuracy = np.sum((predictions["TREND_DIRECTION"] == labels["TREND_DIRECTION"])) / num_predictions
    cumulative_returns = np.cumsum(predictions['RETURN_RATE'])
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = running_max - cumulative_returns
    max_drawdown = np.max(drawdowns) / running_max[np.argmax(drawdowns)]
    volatility = np.std(predictions['RETURN_RATE']) * np.sqrt(num_predictions)
    # TODO: sharp ratio

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'IC': np.mean(IC),
        'ICIR': ICIR,
        'RANK_IC': RANK_IC,
        'RANK_ICIR': RANK_ICIR,
        'directional_accuracy': directional_accuracy,
        'max_drawdown': max_drawdown,
        'volatility': volatility
    }
