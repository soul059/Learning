---
tags:
  - sql
  - database
  - query-language
  - rdbms
aliases:
  - Structured Query Language
---

# SQL (Structured Query Language)

SQL is the standard language for managing data in [[Relational Model|Relational Database Management Systems (RDBMS)]]. It is a powerful, primarily **declarative** language used for querying, manipulating, and defining data.

Proposed in the early 1970s, largely based on concepts from [[Relational Algebra]] and [[Relational Calculus]], SQL has become the industry standard (ANSI and ISO standards exist) for interacting with databases like MySQL, PostgreSQL, Oracle, Microsoft SQL Server, SQLite, and many others.

While often categorized simply as a "query language," SQL encompasses several types of commands used for different purposes:

* **DQL (Data Query Language):** Used for retrieving data from the database. The primary command is `SELECT`. This is often considered part of DML.
* **DML (Data Manipulation Language):** Used for adding, deleting, and modifying data in the database. Commands include `INSERT`, `UPDATE`, and `DELETE`.
* **DDL (Data Definition Language):** Used for defining and modifying the database schema and its objects (tables, views, indexes, etc.). Commands include `CREATE`, `ALTER`, `DROP`, `TRUNCATE`, and `RENAME`.
* **DCL (Data Control Language):** Used for controlling access to data and database objects. Commands include `GRANT` and `REVOKE`.
* **TCL (Transaction Control Language):** Used for managing [[SQL Transactions|transactions]] within the database, ensuring data integrity. Commands include `COMMIT`, `ROLLBACK`, and `SAVEPOINT`.

Understanding SQL is crucial for anyone working with relational databases, from developers and data analysts to database administrators.

---

## Key Concepts in SQL

Explore the fundamental components and operations of SQL:

* [[SQL Data Types]]
* [[SQL DDL]] (Data Definition Language)
* [[SQL DML]] (Data Manipulation Language - Insert, Update, Delete)
* [[SQL DQL]] (Data Query Language - SELECT)
* [[SQL Constraints]]
* [[SQL Joins]]
* [[SQL Aggregate Functions and Grouping]]
* [[SQL Subqueries]]
* [[SQL Views]]
* [[SQL NULL Values]]
* [[SQL Transactions]]

---
_**Related Notes:**_
- [[Relational Model]]
- [[Relational Algebra]]
- [[Relational Calculus]]
- [[DBMS]]
- [[Database Concepts]] (link to your main note if it covers ACID, etc.)