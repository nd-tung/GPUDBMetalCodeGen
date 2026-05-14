-- S18: GROUP BY CHAR1 column
SELECT l_returnflag, l_linestatus, COUNT(*) AS cnt
FROM lineitem
WHERE l_orderkey BETWEEN 1 AND 5000
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus
