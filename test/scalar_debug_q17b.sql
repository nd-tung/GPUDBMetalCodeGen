-- Debug: Q17 pattern on single partkey 1
SELECT l_extendedprice, l_quantity
FROM lineitem, part
WHERE p_partkey = l_partkey
  AND p_partkey = 1
  AND l_quantity < (
    SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey
  )
LIMIT 5
