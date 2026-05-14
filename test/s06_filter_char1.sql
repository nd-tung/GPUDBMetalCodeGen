-- S06: CHAR1 filter (= and IN)
SELECT l_orderkey, l_linenumber, l_returnflag, l_linestatus
FROM lineitem
WHERE l_returnflag = 'R'
  AND l_linestatus IN ('F', 'O')
  AND l_orderkey < 100
ORDER BY l_orderkey, l_linenumber
LIMIT 10
