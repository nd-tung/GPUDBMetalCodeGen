-- S04: IN-list filter on INT
SELECT l_orderkey, l_linenumber
FROM lineitem
WHERE l_linenumber IN (1, 3, 5, 7)
  AND l_orderkey < 50
ORDER BY l_orderkey, l_linenumber
LIMIT 20
