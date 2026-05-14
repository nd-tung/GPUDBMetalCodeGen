-- S10: Arithmetic expressions in SELECT
SELECT 
    l_orderkey,
    l_extendedprice,
    l_discount,
    l_extendedprice * (1 - l_discount) AS disc_price,
    l_extendedprice * (1 - l_discount) * (1 + l_tax) AS charge,
    l_quantity + 10 AS qty_plus_10
FROM lineitem
WHERE l_orderkey BETWEEN 1 AND 10
