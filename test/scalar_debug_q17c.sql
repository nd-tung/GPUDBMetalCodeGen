-- Debug: Q17 pattern with brand/container filter, single partkey
SELECT l_extendedprice, l_quantity
FROM lineitem, part
WHERE p_partkey = l_partkey
  AND p_brand = 'Brand#23'
  AND p_container = 'MED BOX'
  AND l_partkey = 1
  AND l_quantity < (
    SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey
  )
