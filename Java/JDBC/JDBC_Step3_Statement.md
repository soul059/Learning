# JDBC Step 3: Statement Objects - Executing SQL Commands

After establishing a `Connection`, you need a mechanism to send SQL commands to the database. JDBC provides three types of statement interfaces that serve as vehicles for executing your SQL queries, each designed for specific use cases and performance requirements.

---

## Statement Interface Hierarchy

JDBC provides three main types of statement interfaces:

1.  **`Statement`**: For executing simple, static SQL queries
2.  **`PreparedStatement`**: For executing pre-compiled SQL queries with parameters ⭐ **Recommended**
3.  **`CallableStatement`**: For executing stored procedures and functions

```
Statement (interface)
    ↳ PreparedStatement (interface)
        ↳ CallableStatement (interface)
```

---

## 1. Statement Interface - Basic SQL Execution

The `Statement` interface provides the most basic way to execute SQL commands. It sends SQL strings directly to the database for compilation and execution.

### When to Use Statement (Limited Cases)
- Simple, one-time queries with no parameters
- Data Definition Language (DDL) commands like `CREATE TABLE`, `DROP TABLE`
- Administrative commands
- Dynamic SQL where the entire query structure changes

### Critical Limitations ⚠️
- **Security Risk**: Vulnerable to SQL injection attacks
- **Performance**: No pre-compilation benefits
- **Type Safety**: No parameter validation
- **Maintainability**: Harder to read and maintain

### Basic Statement Example

```java
public class StatementExample {
    
    public void createTableExample(Connection conn) throws SQLException {
        String createTableSQL = """
            CREATE TABLE IF NOT EXISTS products (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                price DECIMAL(10,2),
                category VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """;
        
        try (Statement stmt = conn.createStatement()) {
            stmt.executeUpdate(createTableSQL);
            System.out.println("Table 'products' created successfully");
        }
    }
    
    // ❌ DANGEROUS - Vulnerable to SQL injection
    public void unsafeQueryExample(Connection conn, String userInput) throws SQLException {
        String sql = "SELECT * FROM products WHERE name = '" + userInput + "'";
        
        try (Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            
            while (rs.next()) {
                System.out.println("Product: " + rs.getString("name"));
            }
        }
        // If userInput = "'; DROP TABLE products; --"
        // The actual SQL becomes: SELECT * FROM products WHERE name = ''; DROP TABLE products; --'
        // This could delete your entire table!
    }
}
```

---

## 2. PreparedStatement Interface - The Recommended Approach ⭐

`PreparedStatement` extends `Statement` and represents a pre-compiled SQL statement. It's the preferred way to execute SQL in JDBC applications due to its security, performance, and maintainability benefits.

### Key Advantages of PreparedStatement

#### 1. **Prevents SQL Injection** 🔒
The most critical benefit. By using parameter placeholders (`?`), user input is treated as literal data, not executable code.

```java
// ✅ SAFE - Parameters are properly escaped
String sql = "SELECT * FROM users WHERE username = ? AND password = ?";
PreparedStatement pstmt = conn.prepareStatement(sql);
pstmt.setString(1, userInput);  // Automatically escaped
pstmt.setString(2, password);   // Automatically escaped
```

#### 2. **Better Performance** ⚡
SQL is pre-compiled once and can be executed multiple times with different parameters.

```java
// Compiled once, executed many times
String sql = "INSERT INTO logs (user_id, action, timestamp) VALUES (?, ?, ?)";
try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
    
    for (LogEntry entry : logEntries) {
        pstmt.setInt(1, entry.getUserId());
        pstmt.setString(2, entry.getAction());
        pstmt.setTimestamp(3, entry.getTimestamp());
        pstmt.addBatch();  // Add to batch for bulk execution
    }
    
    int[] results = pstmt.executeBatch();  // Execute all at once
}
```

#### 3. **Type Safety and Validation** ✅
Parameters are strongly typed and validated at runtime.

```java
PreparedStatement pstmt = conn.prepareStatement(
    "UPDATE products SET price = ?, last_updated = ? WHERE id = ?"
);

pstmt.setBigDecimal(1, new BigDecimal("29.99"));  // Type-safe decimal
pstmt.setTimestamp(2, Timestamp.valueOf(LocalDateTime.now()));  // Type-safe timestamp
pstmt.setInt(3, productId);  // Type-safe integer
```

#### 4. **Better Readability** 📖
SQL structure is clear and separated from data.

```java
String sql = """
    SELECT p.name, p.price, c.name as category_name 
    FROM products p 
    JOIN categories c ON p.category_id = c.id 
    WHERE p.price BETWEEN ? AND ? 
    AND c.name = ?
    ORDER BY p.price
    """;

try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
    pstmt.setBigDecimal(1, minPrice);
    pstmt.setBigDecimal(2, maxPrice);
    pstmt.setString(3, categoryName);
    
    try (ResultSet rs = pstmt.executeQuery()) {
        // Process results...
    }
}
```

---

## Comprehensive PreparedStatement Examples

### Example 1: CRUD Operations with Error Handling

```java
public class ProductDAO {
    private static final String INSERT_PRODUCT = 
        "INSERT INTO products (name, price, category_id, description) VALUES (?, ?, ?, ?)";
    
    private static final String UPDATE_PRODUCT = 
        "UPDATE products SET name = ?, price = ?, category_id = ?, description = ? WHERE id = ?";
    
    private static final String DELETE_PRODUCT = 
        "DELETE FROM products WHERE id = ?";
    
    private static final String SELECT_PRODUCT_BY_ID = 
        "SELECT id, name, price, category_id, description, created_at FROM products WHERE id = ?";
    
    private static final String SELECT_PRODUCTS_BY_CATEGORY = 
        "SELECT id, name, price, description FROM products WHERE category_id = ? ORDER BY name";
    
    // CREATE
    public long createProduct(Connection conn, Product product) throws SQLException {
        try (PreparedStatement pstmt = conn.prepareStatement(INSERT_PRODUCT, Statement.RETURN_GENERATED_KEYS)) {
            
            pstmt.setString(1, product.getName());
            pstmt.setBigDecimal(2, product.getPrice());
            pstmt.setInt(3, product.getCategoryId());
            pstmt.setString(4, product.getDescription());
            
            int affectedRows = pstmt.executeUpdate();
            
            if (affectedRows == 0) {
                throw new SQLException("Creating product failed, no rows affected.");
            }
            
            try (ResultSet generatedKeys = pstmt.getGeneratedKeys()) {
                if (generatedKeys.next()) {
                    return generatedKeys.getLong(1);
                } else {
                    throw new SQLException("Creating product failed, no ID obtained.");
                }
            }
        }
    }
    
    // READ
    public Optional<Product> findProductById(Connection conn, int productId) throws SQLException {
        try (PreparedStatement pstmt = conn.prepareStatement(SELECT_PRODUCT_BY_ID)) {
            
            pstmt.setInt(1, productId);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    return Optional.of(mapResultSetToProduct(rs));
                } else {
                    return Optional.empty();
                }
            }
        }
    }
    
    // UPDATE
    public boolean updateProduct(Connection conn, Product product) throws SQLException {
        try (PreparedStatement pstmt = conn.prepareStatement(UPDATE_PRODUCT)) {
            
            pstmt.setString(1, product.getName());
            pstmt.setBigDecimal(2, product.getPrice());
            pstmt.setInt(3, product.getCategoryId());
            pstmt.setString(4, product.getDescription());
            pstmt.setInt(5, product.getId());
            
            int affectedRows = pstmt.executeUpdate();
            return affectedRows > 0;
        }
    }
    
    // DELETE
    public boolean deleteProduct(Connection conn, int productId) throws SQLException {
        try (PreparedStatement pstmt = conn.prepareStatement(DELETE_PRODUCT)) {
            
            pstmt.setInt(1, productId);
            
            int affectedRows = pstmt.executeUpdate();
            return affectedRows > 0;
        }
    }
    
    private Product mapResultSetToProduct(ResultSet rs) throws SQLException {
        return new Product(
            rs.getInt("id"),
            rs.getString("name"),
            rs.getBigDecimal("price"),
            rs.getInt("category_id"),
            rs.getString("description"),
            rs.getTimestamp("created_at").toLocalDateTime()
        );
    }
}
```

### Example 2: Complex Queries with Multiple Parameters

```java
public class AdvancedProductQueries {
    
    public List<Product> searchProducts(Connection conn, ProductSearchCriteria criteria) 
            throws SQLException {
        
        StringBuilder sql = new StringBuilder(
            "SELECT p.*, c.name as category_name FROM products p " +
            "JOIN categories c ON p.category_id = c.id WHERE 1=1"
        );
        
        List<Object> parameters = new ArrayList<>();
        
        // Dynamic query building
        if (criteria.getName() != null) {
            sql.append(" AND p.name ILIKE ?");
            parameters.add("%" + criteria.getName() + "%");
        }
        
        if (criteria.getMinPrice() != null) {
            sql.append(" AND p.price >= ?");
            parameters.add(criteria.getMinPrice());
        }
        
        if (criteria.getMaxPrice() != null) {
            sql.append(" AND p.price <= ?");
            parameters.add(criteria.getMaxPrice());
        }
        
        if (criteria.getCategoryIds() != null && !criteria.getCategoryIds().isEmpty()) {
            sql.append(" AND p.category_id IN (");
            sql.append("?,".repeat(criteria.getCategoryIds().size()));
            sql.setLength(sql.length() - 1); // Remove last comma
            sql.append(")");
            parameters.addAll(criteria.getCategoryIds());
        }
        
        sql.append(" ORDER BY ").append(criteria.getSortBy()).append(" ").append(criteria.getSortOrder());
        sql.append(" LIMIT ? OFFSET ?");
        parameters.add(criteria.getLimit());
        parameters.add(criteria.getOffset());
        
        try (PreparedStatement pstmt = conn.prepareStatement(sql.toString())) {
            
            // Set parameters
            for (int i = 0; i < parameters.size(); i++) {
                pstmt.setObject(i + 1, parameters.get(i));
            }
            
            try (ResultSet rs = pstmt.executeQuery()) {
                List<Product> products = new ArrayList<>();
                while (rs.next()) {
                    products.add(mapResultSetToProductWithCategory(rs));
                }
                return products;
            }
        }
    }
}
```

### Example 3: Batch Operations for High Performance

```java
public class BatchOperationExample {
    
    public void insertProductsBatch(Connection conn, List<Product> products) throws SQLException {
        String sql = "INSERT INTO products (name, price, category_id, description) VALUES (?, ?, ?, ?)";
        
        try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            conn.setAutoCommit(false); // Start transaction
            
            int batchSize = 1000;
            int count = 0;
            
            for (Product product : products) {
                pstmt.setString(1, product.getName());
                pstmt.setBigDecimal(2, product.getPrice());
                pstmt.setInt(3, product.getCategoryId());
                pstmt.setString(4, product.getDescription());
                
                pstmt.addBatch();
                count++;
                
                // Execute batch every 1000 records
                if (count % batchSize == 0) {
                    pstmt.executeBatch();
                    pstmt.clearBatch();
                }
            }
            
            // Execute remaining records
            if (count % batchSize != 0) {
                pstmt.executeBatch();
            }
            
            conn.commit(); // Commit transaction
            System.out.println("Successfully inserted " + products.size() + " products");
            
        } catch (SQLException e) {
            conn.rollback(); // Rollback on error
            throw new SQLException("Batch insert failed", e);
        } finally {
            conn.setAutoCommit(true); // Restore auto-commit
        }
    }
}
```

---

## Parameter Setting Methods

### Common setXxx() Methods

| Method | Java Type | SQL Type | Example |
|--------|-----------|----------|---------|
| `setString()` | String | VARCHAR, TEXT | `pstmt.setString(1, "John Doe")` |
| `setInt()` | int | INTEGER | `pstmt.setInt(2, 25)` |
| `setLong()` | long | BIGINT | `pstmt.setLong(3, 1234567890L)` |
| `setBigDecimal()` | BigDecimal | DECIMAL, NUMERIC | `pstmt.setBigDecimal(4, new BigDecimal("99.99"))` |
| `setBoolean()` | boolean | BOOLEAN | `pstmt.setBoolean(5, true)` |
| `setDate()` | java.sql.Date | DATE | `pstmt.setDate(6, Date.valueOf(LocalDate.now()))` |
| `setTimestamp()` | java.sql.Timestamp | TIMESTAMP | `pstmt.setTimestamp(7, Timestamp.valueOf(LocalDateTime.now()))` |
| `setBytes()` | byte[] | BYTEA, BLOB | `pstmt.setBytes(8, imageData)` |

### Handling Null Values

```java
// Method 1: Using setNull()
if (product.getDescription() != null) {
    pstmt.setString(4, product.getDescription());
} else {
    pstmt.setNull(4, Types.VARCHAR);
}

// Method 2: Using setObject() (automatically handles nulls)
pstmt.setObject(4, product.getDescription()); // Will set NULL if description is null
```

### Advanced Parameter Types

```java
// JSON data (PostgreSQL)
pstmt.setObject(1, jsonData, Types.OTHER);

// Arrays (PostgreSQL)
String[] tags = {"electronics", "mobile", "smartphone"};
Array sqlArray = conn.createArrayOf("VARCHAR", tags);
pstmt.setArray(2, sqlArray);

// Large objects
try (InputStream inputStream = new FileInputStream("large-file.pdf")) {
    pstmt.setBinaryStream(3, inputStream);
}
```

---

## CallableStatement - Stored Procedures

For executing stored procedures and functions in the database.

```java
public class StoredProcedureExample {
    
    public BigDecimal calculateOrderTotal(Connection conn, int orderId) throws SQLException {
        
        // PostgreSQL function: CREATE FUNCTION calculate_order_total(order_id INTEGER) RETURNS DECIMAL
        String sql = "{? = call calculate_order_total(?)}";
        
        try (CallableStatement cstmt = conn.prepareCall(sql)) {
            
            // Register the output parameter
            cstmt.registerOutParameter(1, Types.DECIMAL);
            
            // Set the input parameter
            cstmt.setInt(2, orderId);
            
            // Execute the stored procedure
            cstmt.execute();
            
            // Get the result
            return cstmt.getBigDecimal(1);
        }
    }
    
    public void updateInventoryAndLog(Connection conn, int productId, int quantity) throws SQLException {
        
        // Stored procedure with multiple parameters
        String sql = "{call update_inventory_and_log(?, ?, ?)}";
        
        try (CallableStatement cstmt = conn.prepareCall(sql)) {
            
            cstmt.setInt(1, productId);
            cstmt.setInt(2, quantity);
            cstmt.setTimestamp(3, Timestamp.valueOf(LocalDateTime.now()));
            
            cstmt.execute();
        }
    }
}
```

---

## Best Practices and Performance Tips

### 1. Always Use PreparedStatement for User Input
```java
// ✅ Good
String sql = "SELECT * FROM users WHERE email = ?";
PreparedStatement pstmt = conn.prepareStatement(sql);
pstmt.setString(1, userEmail);

// ❌ Bad - SQL injection risk
String sql = "SELECT * FROM users WHERE email = '" + userEmail + "'";
Statement stmt = conn.createStatement();
```

### 2. Reuse PreparedStatements When Possible
```java
// ✅ Good - Reuse the same PreparedStatement
String sql = "INSERT INTO logs (message, timestamp) VALUES (?, ?)";
try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
    for (LogEntry entry : entries) {
        pstmt.setString(1, entry.getMessage());
        pstmt.setTimestamp(2, entry.getTimestamp());
        pstmt.addBatch();
    }
    pstmt.executeBatch();
}
```

### 3. Use Batch Operations for Multiple Records
```java
// Process large datasets efficiently
pstmt.addBatch();  // Add to batch
pstmt.executeBatch();  // Execute all at once
pstmt.clearBatch();  // Clear for next batch
```

### 4. Set Fetch Size for Large Result Sets
```java
PreparedStatement pstmt = conn.prepareStatement(sql);
pstmt.setFetchSize(1000);  // Fetch 1000 rows at a time
```

### 5. Use try-with-resources for Automatic Cleanup
```java
try (PreparedStatement pstmt = conn.prepareStatement(sql);
     ResultSet rs = pstmt.executeQuery()) {
    // Process results
} // Automatic cleanup
```

### 6. Handle Exceptions Appropriately
```java
try {
    // Database operations
} catch (SQLException e) {
    logger.error("Database operation failed: " + e.getMessage(), e);
    // Handle specific error codes
    if (e.getErrorCode() == 23505) { // Unique constraint violation
        throw new DuplicateRecordException("Record already exists");
    }
    throw new DatabaseOperationException("Database operation failed", e);
}
```

---

## Common Pitfalls to Avoid

1. **Using Statement instead of PreparedStatement** - Security and performance issues
2. **Not closing resources** - Memory leaks and connection exhaustion
3. **Concatenating user input into SQL** - SQL injection vulnerability
4. **Not using batch operations** - Poor performance for multiple operations
5. **Ignoring SQLException details** - Missing important error information
6. **Not setting appropriate fetch sizes** - Poor performance with large result sets
7. **Not reusing PreparedStatements** - Unnecessary compilation overhead

    **SQL Injection Example (What to AVOID):**
    ```java
    // UNSAFE CODE - VULNERABLE TO SQL INJECTION
    String userId = "105 OR 1=1"; // Malicious input
    String sql = "SELECT * FROM users WHERE id = " + userId;
    Statement stmt = conn.createStatement();
    ResultSet rs = stmt.executeQuery(sql); // This would return all users!
    ```

    **Safe Code with `PreparedStatement`:**
    ```java
    // SAFE CODE
    String sql = "SELECT * FROM users WHERE id = ?";
    PreparedStatement pstmt = conn.prepareStatement(sql);
    pstmt.setString(1, "105 OR 1=1"); // The malicious string is treated as a literal value
    ResultSet rs = pstmt.executeQuery(); // This will safely find no user with that ID
    ```

2.  **Better Performance**: The SQL query is sent to the database and compiled only once. If you execute the same `PreparedStatement` multiple times with different parameters, the database can reuse the pre-compiled execution plan, which is much faster.

3.  **Improved Readability**: Using `?` placeholders makes the SQL query cleaner and easier to read compared to complex string concatenation.

### How to Use `PreparedStatement`

1.  **Create the SQL with placeholders**: Write your SQL query using `?` for any dynamic values.
    ```java
    String sql = "INSERT INTO employees (name, salary) VALUES (?, ?)";
    ```

2.  **Create the `PreparedStatement` object**:
    ```java
    PreparedStatement pstmt = connection.prepareStatement(sql);
    ```

3.  **Set the parameter values**: Use the various `setXxx()` methods (e.g., `setString()`, `setInt()`, `setDouble()`) to bind values to the placeholders. The parameters are 1-indexed.
    ```java
    pstmt.setString(1, "Jane Doe"); // Set the first '?'
    pstmt.setDouble(2, 80000.00);  // Set the second '?'
    ```

4.  **Execute the query**: Call `executeQuery()` or `executeUpdate()`.
    ```java
    int rowsAffected = pstmt.executeUpdate();
    ```

### Full Example

```java
String sql = "UPDATE employees SET salary = ? WHERE id = ?";

try (Connection conn = DriverManager.getConnection(URL, USER, PASS);
     PreparedStatement pstmt = conn.prepareStatement(sql)) {

    // Set the values for the placeholders
    pstmt.setDouble(1, 95000.00); // New salary
    pstmt.setInt(2, 101);         // Employee ID

    int rowsAffected = pstmt.executeUpdate();
    System.out.println(rowsAffected + " row(s) updated successfully.");

} catch (SQLException e) {
    e.printStackTrace();
}
```

**Conclusion:** Always favor `PreparedStatement` over `Statement` to build secure, efficient, and maintainable database applications in Java.
