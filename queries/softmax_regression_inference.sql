WITH w(r, c, val) AS (
    SELECT r, c, val FROM M ORDER BY r
), exp_scores(r, c, val) AS (
    SELECT X.r, w.c, exp(SUM(X.val * w.val) + {bias}) FROM X, w WHERE X.c = w.r GROUP BY X.r, w.c
), sum_exp_scores(r, val) AS ( -- np.sum(exp_scores, axis = 1)
    SELECT exp_scores.r, SUM(exp_scores.val) FROM exp_scores GROUP BY exp_scores.r
), softmax(r, c, val) AS ( -- exp_scores / sum_exp_scores
    SELECT exp_scores.r, exp_scores.c, exp_scores.val / sum_exp_scores.val FROM exp_scores, sum_exp_scores WHERE exp_scores.r = sum_exp_scores.r
), max_softmax(r, val) AS (
    SELECT softmax.r, max(softmax.val) FROM softmax GROUP BY softmax.r
), predicted(r, val) AS (
    SELECT softmax.r, softmax.c FROM softmax, max_softmax WHERE softmax.r = max_softmax.r AND softmax.val = max_softmax.val  
), accuracy(val) AS (
    SELECT SUM(1)::float / (SELECT COUNT(*) FROM y) FROM y, predicted WHERE y.r = predicted.r AND y.val = predicted.val
) SELECT accuracy.val from accuracy;