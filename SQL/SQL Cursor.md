---
tags:
  - sql
  - cursor
  - procedural-sql
  - plsql
  - tsql
---

# SQL Cursor

In [[SQL|SQL]], particularly within procedural extensions like PL/SQL (Oracle), T-SQL (SQL Server), or procedural blocks in other [[DBMS|DBMS]], a **cursor** is a database object used to retrieve data from a result set one row at a time.

Normally, a [[SQL DQL|SELECT]] statement returns an entire result set. However, sometimes you need to process individual rows within that result set sequentially. A cursor provides a mechanism to navigate through the rows of a result set and process each row individually.

## Why Use Cursors?

Cursors are typically used when you need to perform row-by-row operations that cannot be easily achieved with set-based SQL operations. Common use cases include:

* Processing data from a result set into variables.
* Performing complex procedural logic on each row.
* Updating or deleting specific rows in the result set based on procedural conditions.

## Cursor Operations

Working with a cursor involves several steps:

1.  **DECLARE:** Define the cursor, associating it with a [[SQL DQL|SELECT]] statement.
    ```sql
    DECLARE cursor_name CURSOR FOR
    SELECT column1, column2, ...
    FROM table_name
    WHERE condition;
    ```
    *(Syntax varies slightly by DBMS)*

2.  **OPEN:** Execute the `SELECT` statement defined in the cursor declaration and load the result set. This makes the rows available for fetching.
    ```sql
    OPEN cursor_name;
    ```

3.  **FETCH:** Retrieve a single row from the result set and place the column values into variables. The cursor maintains a pointer to the current row; fetching advances the pointer to the next row.
    ```sql
    FETCH NEXT FROM cursor_name INTO @variable1, @variable2, ...;
    ```
    *(Syntax varies; `INTO` clause and variable syntax differ)*

4.  **Process:** Perform the required logic using the data stored in the variables. This happens within a loop.

5.  **CLOSE:** Deactivate the cursor and release the current result set.
    ```sql
    CLOSE cursor_name;
    ```

6.  **DEALLOCATE:** Remove the cursor definition and free associated system resources.
    ```sql
    DEALLOCATE cursor_name;
    ```
    *(Syntax varies; `DEALLOCATE` in T-SQL, implicit in PL/SQL when block ends)*

## Example (Conceptual - T-SQL Syntax)

```sql
-- Declare variables to hold fetched data
DECLARE @studentId INT;
DECLARE @studentName VARCHAR(100);

-- Declare the cursor
DECLARE student_cursor CURSOR FOR
SELECT StudentID, FirstName + ' ' + LastName
FROM Students
WHERE EnrollmentDate %3E= '2024-01-01';

-- Open the cursor
OPEN student_cursor;

-- Fetch the first row
FETCH NEXT FROM student_cursor INTO @studentId, @studentName;

-- Loop through the result set
WHILE @@FETCH_STATUS = 0 -- Check if FETCH was successful
BEGIN
    -- Process the fetched row (example: print values)
    PRINT 'Processing Student ID: ' + CAST(@studentId AS VARCHAR) + ', Name: ' + @studentName;

    -- Fetch the next row
    FETCH NEXT FROM student_cursor INTO @studentId, @studentName;
END

-- Close the cursor
CLOSE student_cursor;

-- Deallocate the cursor
DEALLOCATE student_cursor;
```

## Disadvantages of Cursors

While useful in specific scenarios, cursors should generally be avoided in favor of set-based operations whenever possible due to performance implications:

* **Performance Overhead:** Processing row by row is typically much slower than processing data in sets, which is how relational databases are optimized to work. Cursors involve significant overhead for opening, fetching, and closing.
* **Increased Resource Usage:** Cursors consume server resources (memory, locks) for their duration.
* **Complexity:** Cursor logic is often more complex and harder to read and maintain than equivalent set-based SQL.

**Best Practice:** Always try to solve problems using set-based SQL operations (`SELECT`, `INSERT`, `UPDATE`, `DELETE` with `WHERE` clauses, [[SQL Joins|joins]], [[SQL Aggregate Functions and Grouping|grouping]], [[SQL Subqueries|subqueries]]) before resorting to cursors. Cursors are often a last resort for truly procedural requirements.

---
**Related Notes:**
* [[SQL]]
* [[SQL DQL]]
* [[SQL DML]]
* [[Procedural Programming]] (Link to or create)
* [[DBMS]]