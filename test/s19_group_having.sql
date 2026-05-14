-- S19: GROUP BY with HAVING
SELECT l_linenumber, COUNT(*) AS cnt, SUM(l_quantity) AS total
FROM lineitem
WHERE l_orderkey BETWEEN 1 AND 2000
GROUP BY l_linenumber
HAVING COUNT(*) > 100
ORDER BY l_linenumber
