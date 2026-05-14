-- S02: Equality filter on INT
SELECT l_orderkey, l_linenumber, l_quantity
FROM lineitem
WHERE l_orderkey = 42
ORDER BY l_orderkey, l_linenumber
