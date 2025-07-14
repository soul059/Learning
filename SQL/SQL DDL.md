---
tags:
  - sql
  - ddl
  - database-schema
  - create-table
  - alter-table
  - drop-table
---
# SQL DDL (Data Definition Language)

[[SQL|DDL (Data Definition Language)]] commands are used to define, modify, and drop database objects such as databases, tables, views, indexes, etc. They deal with the structure (schema) of the database, not the data itself.

Common DDL commands include:

* `CREATE`: To create database objects.
* `ALTER`: To modify the definition of database objects.
* `DROP`: To delete database objects.
* `TRUNCATE`: To remove all records from a table efficiently.
* `RENAME`: To rename a database object.

## CREATE

The `CREATE` statement is used to build new database objects.

### CREATE DATABASE

Creates a new database.

```sql
CREATE DATABASE database_name;
```

### CREATE TABLE

Creates a new table within the current database. This is where you define the table's structure, including column names, [[SQL Data Types|data types]], and [[SQL Constraints|constraints]].

```sql
CREATE TABLE table_name (
    column1 datatype constraints,
    column2 datatype constraints,
    column3 datatype constraints,
    ...
);
```

* `columnX datatype`: Defines a column and its data type.
* `constraints`: Optional keywords to define rules for the column or table (e.g., [[SQL Constraints|NOT NULL]], [[SQL Constraints|PRIMARY KEY]], [[SQL Constraints|FOREIGN KEY]]).

**Example:**

```sql
CREATE TABLE Students (
    StudentID INT PRIMARY KEY,
    FirstName VARCHAR(50) NOT NULL,
    LastName VARCHAR(50) NOT NULL,
    DateOfBirth DATE,
    EnrollmentDate DATE DEFAULT CURRENT_DATE,
    MajorID INT,
    FOREIGN KEY (MajorID) REFERENCES Majors(MajorID)
);

CREATE TABLE Majors (
    MajorID INT PRIMARY KEY,
    MajorName VARCHAR(100) UNIQUE
);
```

### CREATE VIEW

Creates a [[SQL Views|view]] (a virtual table based on the result-set of a `SELECT` query).

```sql
CREATE VIEW view_name AS
SELECT column1, column2, ...
FROM table_name
WHERE condition;
```

## ALTER

The `ALTER` statement is used to modify the structure of an existing database object.

### ALTER TABLE

Used to add, delete, or modify columns in an existing table, or to add/drop constraints.

```sql
-- Add a column
ALTER TABLE table_name
ADD column_name datatype constraints;
```

```sql
-- Drop a column
ALTER TABLE table_name
DROP COLUMN column_name;
```

```sql
-- Modify a column (syntax varies by RDBMS)
ALTER TABLE table_name
ALTER COLUMN column_name datatype new_constraints; -- Example (SQL Server/PostgreSQL)
```
```sql
ALTER TABLE table_name
MODIFY COLUMN column_name datatype new_constraints; -- Example (MySQL/Oracle)
```

```sql
-- Add a constraint
ALTER TABLE table_name
ADD constraint_name constraint_type (columns);
```

```sql
-- Drop a constraint (syntax varies)
ALTER TABLE table_name
DROP CONSTRAINT constraint_name; -- Example (Standard SQL)
```
```sql
ALTER TABLE table_name
DROP FOREIGN KEY constraint_name; -- Example (MySQL)
```

**Example:**

```sql
-- Add an Email column to Students table
ALTER TABLE Students
ADD Email VARCHAR(100) UNIQUE;
```

```sql
-- Drop the DateOfBirth column
ALTER TABLE Students
DROP COLUMN DateOfBirth;
```

```sql
-- Add a CHECK constraint
ALTER TABLE Students
ADD CONSTRAINT CK_StudentID_Positive CHECK (StudentID %3E 0);
```

## DROP

The `DROP` statement is used to delete an existing database object. This operation permanently removes the object and all data associated with it.

### DROP DATABASE

Deletes an entire database. **Use with extreme caution!**

```sql
DROP DATABASE database_name;
```

### DROP TABLE

Deletes an entire table. **Use with extreme caution!**

```sql
DROP TABLE table_name;
```

### DROP VIEW

Deletes a view.

```sql
DROP VIEW view_name;
```

## TRUNCATE TABLE

Removes *all* rows from a table, but keeps the table structure. It is generally faster than `DELETE` without a `WHERE` clause because it logs less information and often resets identity columns. It is a DDL command because it's a structural operation (clearing data from the structure), although it affects data.

```sql
TRUNCATE TABLE table_name;
```
*Note: `TRUNCATE TABLE` cannot be rolled back in some RDBMS, unlike `DELETE` within a transaction.*

## RENAME

Used to rename database objects. Syntax varies.

```sql
-- Rename table (Standard SQL)
ALTER TABLE old_table_name RENAME TO new_table_name;
```

```sql
-- Rename table (MySQL)
RENAME TABLE old_table_name TO new_table_name;
```

```sql
-- Rename column (Standard SQL)
ALTER TABLE table_name RENAME COLUMN old_column_name TO new_column_name;
```

DDL commands define the blueprint of your database, working closely with [[SQL Constraints|constraints]] to enforce data rules.

---
**Related Notes:**
* [[SQL]]
* [[SQL Data Types]]
* [[SQL Constraints]]
* [[Database Design]]
* [[Relational Model]]>)