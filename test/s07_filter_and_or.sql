-- S07: AND/OR combinations
SELECT l_orderkey, l_extendedprice, l_discount
FROM lineitem
WHERE (l_orderkey < 20 OR l_orderkey > 6000000)
  AND l_extendedprice > 10000
  AND l_discount > 0.05
LIMIT 10
