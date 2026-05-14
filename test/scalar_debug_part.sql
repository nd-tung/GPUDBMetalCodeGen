-- Test part filter only
SELECT COUNT(*) FROM part
WHERE p_brand = 'Brand#23' AND p_container = 'MED BOX'
