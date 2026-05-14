-- S14: Scalar SUM
SELECT SUM(l_quantity) AS total_qty,
       SUM(l_extendedprice) AS total_price
FROM lineitem
WHERE l_orderkey BETWEEN 1 AND 100
