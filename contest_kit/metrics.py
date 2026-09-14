import numpy as np
def macro_f1(y_true , y_pred , labels = None) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape leechj : {y_true.shape} vs {y_pred.shape}")
    
    if labels is None:
        labels = np.unique(y_true)
    
    f1s = []
    for c in labels :
        true_c = (y_true == c)
        pred_c = (y_pred == c)
    
        tp = int((true_c & pred_c).sum())
        fn = int((~true_c & pred_c).sum())
        fp = int((true_c & ~pred_c).sum())
        
    
        denom = 2 * tp + fp + fn 
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    if not f1s :
        return 0.0
    return float(np.mean(f1s))

def count_score(y_true , y_pred) -> float : 
    y_true = np.asarray(y_true, dtype= float)
    y_pred = np.asarray(y_pred , dtype= float)
    
    if y_true.shape != y_pred.shape :
        raise ValueError(f" Shape lech {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError(f"Khong co gi de cham")
    if (y_true <= 0).any() :
        raise ValueError(f"y_true co phan tu <= 0 , khong chia duoc")
    if(y_pred < 0).any():
        raise ValueError(f"M chet me r con :))")        
            
    mre = float(-(np.abs(y_pred - y_true) / y_true).mean())
    score = float(np.exp(mre))
    
    assert 0.0 <= score <= 1.0 , f"score ngoai khoang : {score}"
    return score 

