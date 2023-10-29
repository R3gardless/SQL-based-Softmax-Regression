WITH RECURSIVE w_init (r, c, val) AS (
  SELECT DISTINCT r, c, 0.0::float FROM weight ORDER BY r
), soft_reg(iterations, r, c, val) AS (
  SELECT 0, r, c, val FROM w_init
  UNION ALL
  SELECT iterations + 1, r, c, val FROM (
    WITH w(iterations, r, c, val) AS (
      SELECT iterations, r, c, val FROM soft_reg
    ), exp_scores(r, c, val) AS ( -- exp_scores = exp(X.dot(w) + C)
      SELECT X.r, w.c, EXP(SUM(X.val * w.val) + {bias}) FROM X, w WHERE X.c = w.r GROUP BY X.r, w.c
    ), sum_exp_scores(r, val) AS ( -- np.sum(exp_scores, axis = 1)
      SELECT exp_scores.r, SUM(exp_scores.val) FROM exp_scores GROUP BY exp_scores.r
    ), softmax(r, c, val) AS ( -- exp_scores / sum_exp_scores
        SELECT exp_scores.r, exp_scores.c, exp_scores.val / sum_exp_scores.val FROM exp_scores, sum_exp_scores WHERE exp_scores.r = sum_exp_scores.r
    ), diff(r, c, val) AS (
        SELECT softmax.r, softmax.c, CASE WHEN y.val = softmax.c THEN softmax.val - 1 ELSE softmax.val END FROM softmax, y WHERE softmax.r = y.r
    ), X_T(r, c, val) AS( 
        SELECT c, r, val FROM X
    ), w_grad(r, c, val) AS (
        SELECT X_T.r, diff.c, (1.0 / {num_outputs}) * SUM(X_T.val * diff.val) FROM X_T, diff WHERE X_T.c = diff.r GROUP BY X_T.r, diff.c
    ) SELECT iterations, w.r, w.c, w.val - {step_width} * w_grad.val FROM w, w_grad WHERE w.r = w_grad.r AND w.c = w_grad.c 
  ) AS w(iterations, r, c, val) WHERE iterations < {iterations}
), w(r, c, val) AS (
    SELECT r, c, val from soft_reg WHERE iterations = (SELECT MAX(iterations) FROM soft_reg)
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
) SELECT w.r, w.c, w.val, accuracy.val from w, accuracy