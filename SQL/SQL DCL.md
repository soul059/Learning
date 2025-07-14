---
tags:
  - sql
  - dcl
  - security
  - permissions
  - authorization
---

# SQL DCL (Data Control Language)

[[SQL|DCL (Data Control Language)]] commands are used to manage access rights and permissions to the data and objects within a database. They deal with **authorization**, determining who can perform specific actions on which database objects. This is a fundamental aspect of [[Data Security]].

The primary DCL commands are:

* `GRANT`: To give users specific privileges.
* `REVOKE`: To remove privileges from users.

These commands are typically used by the [[Database Administrator (DBA)]] or users who have been granted the necessary permissions to manage privileges.

## Database Privileges

A **privilege** is the right to perform a specific action on a specific database object. Common privileges include:

* `SELECT`: Permission to read data from a table or view.
* `INSERT`: Permission to add new rows to a table.
* `UPDATE`: Permission to modify existing rows in a table (can be limited to specific columns).
* `DELETE`: Permission to remove rows from a table.
* `REFERENCES`: Permission to create a [[SQL Constraints|Foreign Key]] constraint that refers to a table.
* `ALL [PRIVILEGES]`: Grants all available privileges on an object.
* Privileges specific to other object types: `EXECUTE` on stored procedures/functions, `ALTER`, `INDEX`, `CREATE`, `DROP`, etc.

Privileges can be granted to individual [[Database Users|users]] or to **roles** (named collections of privileges, which can then be assigned to users, simplifying management).

## GRANT

The `GRANT` statement is used to give specific privileges to a user or role.

```sql
GRANT privilege1, privilege2, ...
ON object_name
TO user_name_or_role_name [WITH GRANT OPTION];
```

* `privilege1, privilege2, ...`: The list of privileges to grant (e.g., `SELECT`, `INSERT`, `UPDATE`, `DELETE`, `ALL`).
* `ON object_name`: The name of the database object the privileges apply to (e.g., a table name, a view name, a procedure name).
* `TO user_name_or_role_name`: The user or role receiving the privileges.
* `[WITH GRANT OPTION]`: (Optional) If specified, the recipient can then grant these same privileges to other users or roles.

**Examples:**

```sql
-- Grant SELECT permission on the Students table to user 'john_doe'
GRANT SELECT
ON Students
TO john_doe;
```

```sql
-- Grant INSERT and UPDATE permissions on the Orders table to role 'order_manager'
GRANT INSERT, UPDATE
ON Orders
TO order_manager;
```

```sql
-- Grant ALL privileges on the Products table to user 'admin_user' with the ability to grant them further
GRANT ALL PRIVILEGES
ON Products
TO admin_user WITH GRANT OPTION;
```

```sql
-- Grant SELECT permission on specific columns of the Employees table
GRANT SELECT (EmployeeID, FirstName, LastName, Department)
ON Employees
TO public_users_role; -- 'public' or similar might be a built-in role for all users
```

## REVOKE

The `REVOKE` statement is used to take back privileges that were previously granted to a user or role.

```sql
REVOKE privilege1, privilege2, ...
ON object_name
FROM user_name_or_role_name [CASCADE | RESTRICT];
```

* `privilege1, privilege2, ...`: The list of privileges to revoke.
* `ON object_name`: The database object.
* `FROM user_name_or_role_name`: The user or role from whom privileges are being removed.
* `[CASCADE | RESTRICT]`: (Optional) Specifies how to handle privileges that were granted to others by the user/role using the `WITH GRANT OPTION`.
    * `CASCADE`: Revokes the privilege from the specified user/role *and* also from anyone they granted it to (and anyone those users granted it to, and so on).
    * `RESTRICT`: If the privilege has been granted to others by the specified user/role, the `REVOKE` statement fails.

**Examples:**

```sql
-- Revoke INSERT permission on the Orders table from role 'order_manager'
REVOKE INSERT
ON Orders
FROM order_manager;
```

```sql
-- Revoke SELECT permission on the Students table from user 'john_doe'
REVOKE SELECT
ON Students
FROM john_doe;
```

```sql
-- Revoke ALL privileges on Products from admin_user, also revoking from anyone they granted them to
REVOKE ALL PRIVILEGES
ON Products
FROM admin_user CASCADE;
```

DCL commands are powerful tools for implementing the [[Database Concepts#Increased Data Security|security policy]] of an organization within the [[DBMS]], controlling precisely who can do what with the data.

---
**Related Notes:**
* [[SQL]]
* [[Database Security]]
* [[Database Administrator (DBA)]]
* [[Database Users]]
* [[SQL DQL]]
* [[SQL DML]]
* [[SQL DDL]]
```

```markdown
***
**End of File: `SQL DCL.md`**
***
```

---

This file provides a detailed explanation of SQL DCL commands. It includes syntax and examples for `GRANT` and `REVOKE`, and links to related concepts. Let me know if you need anything else!