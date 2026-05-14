-- S15: Scalar AVG
SELECT AVG(l_quantity) AS avg_qty,
       AVG(l_extendedprice) AS avg_price,
       AVG(l_discount) AS avg_disc
FROM lineitem
WHERE l_orderkey BETWEEN 1 AND 1000
