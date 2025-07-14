---
tags:
  - sql
  - function
  - stored-function
  - procedural-sql
---

# SQL Function

In [[SQL|SQL]], a **function** (also called a **stored function** or **user-defined function - UDF**) is a type of stored routine that performs a computation and returns a single value. Functions are typically used to encapsulate reusable logic that can be called within [[SQL DQL|SELECT]] statements, [[SQL DML|WHERE]] clauses, or other SQL statements where an expression is expected.

Functions are defined using [[SQL DDL]] statements.

## Purpose of Functions

SQL functions are used to:

* **Encapsulate Reusable Logic:** Write a piece of logic once and call it from multiple queries or procedures.
* **Simplify Complex Calculations:** Hide complex calculations behind a simple function call.
* **Improve Readability:** Make queries easier to understand by replacing complex expressions with descriptive function names.
* **Enhance Data Transformation:** Perform data formatting or transformation within queries.

## Function Characteristics

* **Return Value:** Must return a single value. The data type of the return value is specified when the function is created.
* **Parameters:** Can accept zero or more input parameters.
* **Usage:** Can be used in `SELECT` lists, `WHERE` clauses, `HAVING` clauses, `ORDER BY` clauses, and other places where expressions are valid.
* **No Side Effects (Ideally):** Ideally, functions should be **deterministic** (always return the same output for the same input) and should **not** perform data modifications (`INSERT`, `UPDATE`, `DELETE`). While some [[DBMS|DBMS]] allow functions with side effects, it's generally considered bad practice as it can lead to unpredictable query results and hinder [[Query Optimization|query optimization]].

## Function Syntax (Conceptual)

The exact syntax varies between [[DBMS|Database Management Systems]].

```sql
CREATE FUNCTION function_name (@parameter1 datatype, @parameter2 datatype, ...)
RETURNS return_datatype
AS
BEGIN
    -- Declare variables (optional)
    -- Perform calculations or logic
    -- RETURN value;
END;
```

* `function_name`: A unique name for the function.
* `@parameterX datatype`: Input parameters (syntax varies, e.g., `parameter_name datatype` or `@parameter_name datatype`).
* `RETURNS return_datatype`: Specifies the data type of the single value the function will return.
* `AS BEGIN...END`: Contains the body of the function, including the logic and a `RETURN` statement.

## Example (Conceptual - T-SQL Syntax)

```sql
-- Create a function to calculate the net price after a discount
CREATE FUNCTION CalculateNetPrice (@price DECIMAL(10, 2), @discount DECIMAL(10, 2))
RETURNS DECIMAL(10, 2)
AS
BEGIN
    DECLARE @netPrice DECIMAL(10, 2);

    -- Ensure discount is not negative and not more than price
    IF @discount %3C 0 SET @discount = 0;
    IF @discount %3E @price SET @discount = @price;

    SET @netPrice = @price - @discount;

    RETURN @netPrice;
END;
```

## Using a Function

Once created, you can call the function in your SQL queries.

```sql
-- Select product name, list price, discount, and calculated net price
SELECT
    ProductName,
    Price,
    Discount,
    dbo.CalculateNetPrice(Price, Discount) AS NetPrice -- Call the function
FROM Products;

-- Find products whose net price after discount is less than 50
SELECT ProductName, Price, Discount
FROM Products
WHERE dbo.CalculateNetPrice(Price, Discount) %3C 50; -- Use function in WHERE clause
```
*(`dbo.` is the schema name in SQL Server; schema usage varies)*

## Scalar Functions vs. Table-Valued Functions

The type of function described above is a **Scalar Function** because it returns a single scalar value.

Some [[DBMS|DBMS]] also support **Table-Valued Functions (TVFs)**, which return an entire result set (a table) instead of a single scalar value. TVFs can often be used in the `FROM` clause of a `SELECT` statement, similar to a table or [[SQL Views|view]].

## Functions vs. Stored Procedures

Functions and [[Stored Procedures|stored procedures]] are both types of stored routines, but they have key differences:

| Feature          | SQL Function                                  | Stored Procedure                              |
| :--------------- | :-------------------------------------------- | :-------------------------------------------- |
| **Return Value** | Must return a single value.                   | Can return zero or multiple values (via OUTPUT parameters or result sets). |
| **Usage** | Can be used within SQL statements (SELECT, WHERE, etc.) as expressions. | Called as a separate statement (`EXECUTE` or `CALL`). |
| **Side Effects** | Ideally, no side effects (no data modification). | Can perform data modifications (INSERT, UPDATE, DELETE). |
| **Purpose** | Calculations, data transformation, encapsulating logic for use *within* queries. | Performing actions, executing sequences of SQL statements, complex procedural logic, data modification. |

Functions are best suited for computations and returning a value, while stored procedures are better for executing a sequence of operations or performing actions.

---
**Related Notes:**
* [[SQL]]
* [[SQL DDL]]
* [[SQL DQL]]
* [[SQL DML]]
* [[Stored Procedures]] (Create if needed)
* [[Procedural Programming]] (Link to or create)
* [[Query Optimization]]