-- S17: GROUP BY integer column
SELECT l_linenumber, COUNT(*) AS cnt, SUM(l_quantity) AS total_qty
FROM lineitem
WHERE l_orderkey BETWEEN 1 AND 1000
GROUP BY l_linenumber
ORDER BY l_linenumber
