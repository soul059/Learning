# Modern JDBC Patterns and Best Practices

This guide covers modern patterns, best practices, and advanced techniques for working with JDBC in contemporary Java applications. It focuses on practical, production-ready code that emphasizes security, performance, and maintainability.

---

## Table of Contents

1. [Connection Pooling and Configuration](#connection-pooling-and-configuration)
2. [DAO Pattern Implementation](#dao-pattern-implementation)
3. [Modern Java Features with JDBC](#modern-java-features-with-jdbc)
4. [Error Handling Strategies](#error-handling-strategies)
5. [Performance Optimization](#performance-optimization)
6. [Testing JDBC Code](#testing-jdbc-code)
7. [Security Best Practices](#security-best-practices)

---

## Connection Pooling and Configuration

### HikariCP - The Fastest Connection Pool

HikariCP is the recommended connection pool for modern Java applications due to its excellent performance and reliability.

**Maven Dependency:**
```xml
<dependency>
    <groupId>com.zaxxer</groupId>
    <artifactId>HikariCP</artifactId>
    <version>5.1.0</version>
</dependency>
```

**Configuration Class:**
```java
@Configuration
public class DatabaseConfig {
    
    @Bean
    @Primary
    public DataSource primaryDataSource() {
        HikariConfig config = new HikariConfig();
        
        // Basic connection settings
        config.setJdbcUrl("jdbc:postgresql://localhost:5432/myapp");
        config.setUsername("${db.username}");
        config.setPassword("${db.password}");
        config.setDriverClassName("org.postgresql.Driver");
        
        // Pool settings
        config.setMaximumPoolSize(20);           // Maximum connections in pool
        config.setMinimumIdle(5);                // Minimum idle connections
        config.setConnectionTimeout(30000);      // 30 seconds to get connection
        config.setIdleTimeout(600000);           // 10 minutes idle timeout
        config.setMaxLifetime(1800000);          // 30 minutes max lifetime
        config.setLeakDetectionThreshold(60000); // 1 minute leak detection
        
        // Performance settings
        config.setConnectionTestQuery("SELECT 1");
        config.setValidationTimeout(5000);       // 5 seconds validation timeout
        
        // Connection properties
        config.addDataSourceProperty("cachePrepStmts", "true");
        config.addDataSourceProperty("prepStmtCacheSize", "256");
        config.addDataSourceProperty("prepStmtCacheSqlLimit", "2048");
        config.addDataSourceProperty("useServerPrepStmts", "true");
        config.addDataSourceProperty("rewriteBatchedStatements", "true");
        
        // Pool name for monitoring
        config.setPoolName("MyApp-Primary-Pool");
        
        return new HikariDataSource(config);
    }
    
    @Bean
    public JdbcTemplate jdbcTemplate(@Qualifier("primaryDataSource") DataSource dataSource) {
        return new JdbcTemplate(dataSource);
    }
}
```

**Environment-Specific Configuration:**
```yaml
# application.yml
spring:
  datasource:
    hikari:
      jdbc-url: jdbc:postgresql://localhost:5432/myapp
      username: ${DB_USERNAME:myuser}
      password: ${DB_PASSWORD:mypass}
      maximum-pool-size: ${DB_POOL_SIZE:20}
      minimum-idle: ${DB_MIN_IDLE:5}
      connection-timeout: 30000
      idle-timeout: 600000
      max-lifetime: 1800000
      leak-detection-threshold: 60000
      data-source-properties:
        cachePrepStmts: true
        prepStmtCacheSize: 256
        prepStmtCacheSqlLimit: 2048
        useServerPrepStmts: true
        rewriteBatchedStatements: true
```

---

## DAO Pattern Implementation

### Modern DAO with Generic Base Class

```java
@Repository
public abstract class BaseDAO<T, ID> {
    
    protected final DataSource dataSource;
    protected final RowMapper<T> rowMapper;
    protected final String tableName;
    
    protected BaseDAO(DataSource dataSource, RowMapper<T> rowMapper, String tableName) {
        this.dataSource = dataSource;
        this.rowMapper = rowMapper;
        this.tableName = tableName;
    }
    
    // Generic CRUD operations
    public Optional<T> findById(ID id) {
        String sql = "SELECT * FROM " + tableName + " WHERE id = ?";
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            pstmt.setObject(1, id);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    return Optional.of(rowMapper.mapRow(rs, 1));
                }
                return Optional.empty();
            }
        } catch (SQLException e) {
            throw new DataAccessException("Failed to find entity by id: " + id, e);
        }
    }
    
    public List<T> findAll() {
        String sql = "SELECT * FROM " + tableName + " ORDER BY id";
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql);
             ResultSet rs = pstmt.executeQuery()) {
            
            List<T> results = new ArrayList<>();
            int rowNum = 0;
            while (rs.next()) {
                results.add(rowMapper.mapRow(rs, ++rowNum));
            }
            return results;
            
        } catch (SQLException e) {
            throw new DataAccessException("Failed to find all entities", e);
        }
    }
    
    public long count() {
        String sql = "SELECT COUNT(*) FROM " + tableName;
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql);
             ResultSet rs = pstmt.executeQuery()) {
            
            rs.next();
            return rs.getLong(1);
            
        } catch (SQLException e) {
            throw new DataAccessException("Failed to count entities", e);
        }
    }
    
    // Abstract methods for specific implementations
    public abstract ID save(T entity);
    public abstract boolean update(T entity);
    public abstract boolean deleteById(ID id);
}
```

### Specific DAO Implementation

```java
@Repository
public class EmployeeDAO extends BaseDAO<Employee, Integer> {
    
    private static final String INSERT_SQL = """
        INSERT INTO employees (name, email, salary, department_id, hire_date) 
        VALUES (?, ?, ?, ?, ?)
        """;
    
    private static final String UPDATE_SQL = """
        UPDATE employees 
        SET name = ?, email = ?, salary = ?, department_id = ? 
        WHERE id = ?
        """;
    
    private static final String DELETE_SQL = "DELETE FROM employees WHERE id = ?";
    
    private static final String FIND_BY_DEPARTMENT_SQL = """
        SELECT * FROM employees 
        WHERE department_id = ? 
        ORDER BY name
        """;
    
    public EmployeeDAO(DataSource dataSource) {
        super(dataSource, new EmployeeRowMapper(), "employees");
    }
    
    @Override
    public Integer save(Employee employee) {
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(INSERT_SQL, Statement.RETURN_GENERATED_KEYS)) {
            
            pstmt.setString(1, employee.getName());
            pstmt.setString(2, employee.getEmail());
            pstmt.setBigDecimal(3, employee.getSalary());
            pstmt.setInt(4, employee.getDepartmentId());
            pstmt.setDate(5, Date.valueOf(employee.getHireDate()));
            
            int affectedRows = pstmt.executeUpdate();
            if (affectedRows == 0) {
                throw new DataAccessException("Creating employee failed, no rows affected");
            }
            
            try (ResultSet generatedKeys = pstmt.getGeneratedKeys()) {
                if (generatedKeys.next()) {
                    return generatedKeys.getInt(1);
                } else {
                    throw new DataAccessException("Creating employee failed, no ID obtained");
                }
            }
            
        } catch (SQLException e) {
            throw new DataAccessException("Failed to save employee", e);
        }
    }
    
    @Override
    public boolean update(Employee employee) {
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(UPDATE_SQL)) {
            
            pstmt.setString(1, employee.getName());
            pstmt.setString(2, employee.getEmail());
            pstmt.setBigDecimal(3, employee.getSalary());
            pstmt.setInt(4, employee.getDepartmentId());
            pstmt.setInt(5, employee.getId());
            
            return pstmt.executeUpdate() > 0;
            
        } catch (SQLException e) {
            throw new DataAccessException("Failed to update employee", e);
        }
    }
    
    @Override
    public boolean deleteById(Integer id) {
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(DELETE_SQL)) {
            
            pstmt.setInt(1, id);
            return pstmt.executeUpdate() > 0;
            
        } catch (SQLException e) {
            throw new DataAccessException("Failed to delete employee", e);
        }
    }
    
    // Domain-specific methods
    public List<Employee> findByDepartment(int departmentId) {
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(FIND_BY_DEPARTMENT_SQL)) {
            
            pstmt.setInt(1, departmentId);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                List<Employee> employees = new ArrayList<>();
                int rowNum = 0;
                while (rs.next()) {
                    employees.add(rowMapper.mapRow(rs, ++rowNum));
                }
                return employees;
            }
            
        } catch (SQLException e) {
            throw new DataAccessException("Failed to find employees by department", e);
        }
    }
    
    public List<Employee> findBySalaryRange(BigDecimal minSalary, BigDecimal maxSalary) {
        String sql = """
            SELECT * FROM employees 
            WHERE salary BETWEEN ? AND ? 
            ORDER BY salary DESC
            """;
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            pstmt.setBigDecimal(1, minSalary);
            pstmt.setBigDecimal(2, maxSalary);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                List<Employee> employees = new ArrayList<>();
                int rowNum = 0;
                while (rs.next()) {
                    employees.add(rowMapper.mapRow(rs, ++rowNum));
                }
                return employees;
            }
            
        } catch (SQLException e) {
            throw new DataAccessException("Failed to find employees by salary range", e);
        }
    }
    
    // Row mapper for Employee
    private static class EmployeeRowMapper implements RowMapper<Employee> {
        @Override
        public Employee mapRow(ResultSet rs, int rowNum) throws SQLException {
            return Employee.builder()
                .id(rs.getInt("id"))
                .name(rs.getString("name"))
                .email(rs.getString("email"))
                .salary(rs.getBigDecimal("salary"))
                .departmentId(rs.getInt("department_id"))
                .hireDate(rs.getDate("hire_date").toLocalDate())
                .createdAt(rs.getTimestamp("created_at").toLocalDateTime())
                .build();
        }
    }
}
```

---

## Modern Java Features with JDBC

### Using Records for Data Transfer

```java
// Immutable data record
public record EmployeeView(
    int id,
    String name,
    String email,
    BigDecimal salary,
    String departmentName,
    LocalDate hireDate
) {
    // Custom constructor with validation
    public EmployeeView {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Name cannot be blank");
        }
        if (salary != null && salary.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("Salary cannot be negative");
        }
    }
    
    // Static factory method
    public static EmployeeView fromResultSet(ResultSet rs) throws SQLException {
        return new EmployeeView(
            rs.getInt("id"),
            rs.getString("name"),
            rs.getString("email"),
            rs.getBigDecimal("salary"),
            rs.getString("department_name"),
            rs.getDate("hire_date").toLocalDate()
        );
    }
}
```

### Stream-Based Result Processing

```java
public class StreamBasedDAO {
    
    private final DataSource dataSource;
    
    public Stream<Employee> streamAllEmployees() {
        try {
            Connection conn = dataSource.getConnection();
            PreparedStatement pstmt = conn.prepareStatement(
                "SELECT * FROM employees ORDER BY id",
                ResultSet.TYPE_FORWARD_ONLY,
                ResultSet.CONCUR_READ_ONLY
            );
            pstmt.setFetchSize(1000); // Stream in chunks
            
            ResultSet rs = pstmt.executeQuery();
            
            return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(
                    new ResultSetIterator<>(rs, Employee::fromResultSet),
                    Spliterator.ORDERED
                ),
                false
            ).onClose(() -> {
                try {
                    rs.close();
                    pstmt.close();
                    conn.close();
                } catch (SQLException e) {
                    throw new UncheckedSQLException(e);
                }
            });
            
        } catch (SQLException e) {
            throw new DataAccessException("Failed to stream employees", e);
        }
    }
    
    // Usage example
    public List<String> getHighEarnerNames(BigDecimal threshold) {
        try (Stream<Employee> employeeStream = streamAllEmployees()) {
            return employeeStream
                .filter(emp -> emp.getSalary().compareTo(threshold) > 0)
                .map(Employee::getName)
                .sorted()
                .collect(Collectors.toList());
        }
    }
}

// Helper iterator for ResultSet streaming
class ResultSetIterator<T> implements Iterator<T> {
    private final ResultSet rs;
    private final ResultSetMapper<T> mapper;
    private Boolean hasNext;
    
    public ResultSetIterator(ResultSet rs, ResultSetMapper<T> mapper) {
        this.rs = rs;
        this.mapper = mapper;
    }
    
    @Override
    public boolean hasNext() {
        if (hasNext == null) {
            try {
                hasNext = rs.next();
            } catch (SQLException e) {
                throw new UncheckedSQLException(e);
            }
        }
        return hasNext;
    }
    
    @Override
    public T next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        try {
            T result = mapper.map(rs);
            hasNext = null; // Reset for next iteration
            return result;
        } catch (SQLException e) {
            throw new UncheckedSQLException(e);
        }
    }
}

@FunctionalInterface
interface ResultSetMapper<T> {
    T map(ResultSet rs) throws SQLException;
}
```

### Optional-Based Null Handling

```java
public class OptionalBasedDAO {
    
    public Optional<Employee> findEmployeeByEmail(String email) {
        String sql = "SELECT * FROM employees WHERE email = ?";
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            pstmt.setString(1, email);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                return rs.next() 
                    ? Optional.of(Employee.fromResultSet(rs))
                    : Optional.empty();
            }
            
        } catch (SQLException e) {
            throw new DataAccessException("Failed to find employee by email", e);
        }
    }
    
    // Safe null handling in mapping
    public Employee mapRowWithOptionals(ResultSet rs) throws SQLException {
        return Employee.builder()
            .id(rs.getInt("id"))
            .name(rs.getString("name"))
            .email(rs.getString("email"))
            .salary(Optional.ofNullable(rs.getBigDecimal("salary"))
                        .orElse(BigDecimal.ZERO))
            .departmentId(rs.getInt("department_id"))
            .phone(Optional.ofNullable(rs.getString("phone"))
                       .filter(s -> !s.isBlank()))
            .hireDate(Optional.ofNullable(rs.getDate("hire_date"))
                          .map(Date::toLocalDate)
                          .orElse(LocalDate.now()))
            .build();
    }
}
```

---

## Error Handling Strategies

### Custom Exception Hierarchy

```java
// Base exception for all data access errors
public class DataAccessException extends RuntimeException {
    private final String operation;
    private final Map<String, Object> context;
    
    public DataAccessException(String message, Throwable cause) {
        this(message, cause, null, Map.of());
    }
    
    public DataAccessException(String message, Throwable cause, String operation, Map<String, Object> context) {
        super(message, cause);
        this.operation = operation;
        this.context = Map.copyOf(context);
    }
    
    public Optional<String> getOperation() {
        return Optional.ofNullable(operation);
    }
    
    public Map<String, Object> getContext() {
        return context;
    }
}

// Specific exceptions
public class EntityNotFoundException extends DataAccessException {
    public EntityNotFoundException(String entityType, Object id) {
        super(String.format("%s with id %s not found", entityType, id), null, "FIND", 
              Map.of("entityType", entityType, "id", id));
    }
}

public class DuplicateEntityException extends DataAccessException {
    public DuplicateEntityException(String entityType, String field, Object value) {
        super(String.format("%s with %s '%s' already exists", entityType, field, value), null, "CREATE",
              Map.of("entityType", entityType, "field", field, "value", value));
    }
}

public class ConstraintViolationException extends DataAccessException {
    public ConstraintViolationException(String constraint, Throwable cause) {
        super("Constraint violation: " + constraint, cause, "CONSTRAINT_CHECK",
              Map.of("constraint", constraint));
    }
}
```

### Error Handler with Retry Logic

```java
@Component
public class DatabaseErrorHandler {
    
    private static final Logger logger = LoggerFactory.getLogger(DatabaseErrorHandler.class);
    
    private final RetryTemplate retryTemplate;
    
    public DatabaseErrorHandler() {
        this.retryTemplate = RetryTemplate.builder()
            .maxAttempts(3)
            .exponentialBackoff(1000, 2, 10000)
            .retryOn(SQLTransientException.class, SQLTimeoutException.class)
            .build();
    }
    
    public <T> T executeWithRetry(String operation, Supplier<T> databaseOperation) {
        return retryTemplate.execute(context -> {
            try {
                return databaseOperation.get();
            } catch (SQLException e) {
                logger.warn("Database operation '{}' failed on attempt {}: {}", 
                           operation, context.getRetryCount() + 1, e.getMessage());
                throw handleSQLException(e, operation);
            }
        });
    }
    
    public DataAccessException handleSQLException(SQLException e, String operation) {
        String sqlState = e.getSQLState();
        int errorCode = e.getErrorCode();
        
        // PostgreSQL specific error codes
        return switch (sqlState) {
            case "23505" -> new DuplicateEntityException("Entity", "key", "value");
            case "23503" -> new ConstraintViolationException("Foreign key constraint", e);
            case "23514" -> new ConstraintViolationException("Check constraint", e);
            case "42P01" -> new DataAccessException("Table does not exist", e, operation, Map.of());
            case "28P01" -> new DataAccessException("Authentication failed", e, operation, Map.of());
            case "08006" -> new DataAccessException("Connection failure", e, operation, Map.of());
            default -> new DataAccessException("Database error: " + e.getMessage(), e, operation, 
                                              Map.of("sqlState", sqlState, "errorCode", errorCode));
        };
    }
}
```

---

## Performance Optimization

### Prepared Statement Caching

```java
@Component
public class CachingStatementManager {
    
    private final ConcurrentHashMap<String, PreparedStatement> statementCache = new ConcurrentHashMap<>();
    private final DataSource dataSource;
    private final ScheduledExecutorService cleanupExecutor;
    
    public CachingStatementManager(DataSource dataSource) {
        this.dataSource = dataSource;
        this.cleanupExecutor = Executors.newSingleThreadScheduledExecutor();
        
        // Cleanup cache every hour
        cleanupExecutor.scheduleAtFixedRate(this::cleanupCache, 1, 1, TimeUnit.HOURS);
    }
    
    public PreparedStatement getCachedStatement(Connection conn, String sql) throws SQLException {
        return statementCache.computeIfAbsent(sql, key -> {
            try {
                PreparedStatement stmt = conn.prepareStatement(sql);
                // Configure statement for optimal performance
                stmt.setFetchSize(1000);
                stmt.setQueryTimeout(300); // 5 minutes
                return stmt;
            } catch (SQLException e) {
                throw new UncheckedSQLException(e);
            }
        });
    }
    
    private void cleanupCache() {
        logger.info("Cleaning up statement cache. Current size: {}", statementCache.size());
        
        statementCache.entrySet().removeIf(entry -> {
            try {
                PreparedStatement stmt = entry.getValue();
                if (stmt.isClosed()) {
                    return true;
                }
                // Check if connection is still valid
                return !stmt.getConnection().isValid(5);
            } catch (SQLException e) {
                logger.warn("Error checking statement validity: {}", e.getMessage());
                return true;
            }
        });
    }
    
    @PreDestroy
    public void shutdown() {
        cleanupExecutor.shutdown();
        statementCache.values().forEach(stmt -> {
            try {
                stmt.close();
            } catch (SQLException e) {
                logger.warn("Error closing cached statement: {}", e.getMessage());
            }
        });
    }
}
```

### Batch Processing Optimization

```java
@Service
public class OptimizedBatchProcessor {
    
    private static final int DEFAULT_BATCH_SIZE = 1000;
    private static final int COMMIT_INTERVAL = 5000;
    
    private final DataSource dataSource;
    
    public <T> BatchResult processBatch(List<T> items, BatchProcessor<T> processor) {
        if (items.isEmpty()) {
            return new BatchResult(0, 0, List.of());
        }
        
        long startTime = System.currentTimeMillis();
        List<String> errors = new ArrayList<>();
        int processed = 0;
        int failed = 0;
        
        try (Connection conn = dataSource.getConnection()) {
            conn.setAutoCommit(false);
            
            String sql = processor.getSql();
            try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
                
                for (int i = 0; i < items.size(); i++) {
                    try {
                        processor.setParameters(pstmt, items.get(i));
                        pstmt.addBatch();
                        processed++;
                        
                        // Execute batch when reaching batch size
                        if ((i + 1) % DEFAULT_BATCH_SIZE == 0) {
                            executeBatch(pstmt, errors);
                        }
                        
                        // Commit periodically to avoid long transactions
                        if ((i + 1) % COMMIT_INTERVAL == 0) {
                            conn.commit();
                        }
                        
                    } catch (Exception e) {
                        errors.add("Item " + i + ": " + e.getMessage());
                        failed++;
                    }
                }
                
                // Execute remaining items
                if (processed % DEFAULT_BATCH_SIZE != 0) {
                    executeBatch(pstmt, errors);
                }
                
                conn.commit();
                
            } catch (SQLException e) {
                conn.rollback();
                throw new DataAccessException("Batch processing failed", e);
            }
            
        } catch (SQLException e) {
            throw new DataAccessException("Failed to process batch", e);
        }
        
        long duration = System.currentTimeMillis() - startTime;
        logger.info("Batch processing completed: {} processed, {} failed, {}ms", 
                   processed, failed, duration);
        
        return new BatchResult(processed, failed, errors);
    }
    
    private void executeBatch(PreparedStatement pstmt, List<String> errors) throws SQLException {
        try {
            int[] results = pstmt.executeBatch();
            pstmt.clearBatch();
            
            // Check for failed operations
            for (int i = 0; i < results.length; i++) {
                if (results[i] == PreparedStatement.EXECUTE_FAILED) {
                    errors.add("Batch item " + i + " failed to execute");
                }
            }
        } catch (BatchUpdateException e) {
            // Handle partial batch failures
            int[] updateCounts = e.getUpdateCounts();
            for (int i = 0; i < updateCounts.length; i++) {
                if (updateCounts[i] == PreparedStatement.EXECUTE_FAILED) {
                    errors.add("Batch item " + i + " failed: " + e.getMessage());
                }
            }
            pstmt.clearBatch();
        }
    }
    
    @FunctionalInterface
    public interface BatchProcessor<T> {
        String getSql();
        void setParameters(PreparedStatement pstmt, T item) throws SQLException;
    }
    
    public record BatchResult(int processed, int failed, List<String> errors) {}
}
```

---

## Testing JDBC Code

### Test Database Configuration

```java
@TestConfiguration
public class TestDatabaseConfig {
    
    @Bean
    @Primary
    public DataSource testDataSource() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:h2:mem:testdb;DB_CLOSE_DELAY=-1;MODE=PostgreSQL");
        config.setUsername("sa");
        config.setPassword("");
        config.setMaximumPoolSize(5);
        config.setConnectionTimeout(5000);
        return new HikariDataSource(config);
    }
    
    @Bean
    public DatabaseInitializer databaseInitializer(@Qualifier("testDataSource") DataSource dataSource) {
        return new DatabaseInitializer(dataSource);
    }
    
    @Component
    public static class DatabaseInitializer {
        private final DataSource dataSource;
        
        public DatabaseInitializer(DataSource dataSource) {
            this.dataSource = dataSource;
            initializeSchema();
        }
        
        private void initializeSchema() {
            try (Connection conn = dataSource.getConnection();
                 Statement stmt = conn.createStatement()) {
                
                // Create test tables
                stmt.execute("""
                    CREATE TABLE departments (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        location VARCHAR(255)
                    )
                    """);
                
                stmt.execute("""
                    CREATE TABLE employees (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        salary DECIMAL(12,2),
                        department_id INTEGER REFERENCES departments(id),
                        hire_date DATE DEFAULT CURRENT_DATE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """);
                
                // Insert test data
                stmt.execute("""
                    INSERT INTO departments (name, location) VALUES 
                    ('Engineering', 'San Francisco'),
                    ('Marketing', 'New York'),
                    ('Sales', 'Chicago')
                    """);
                
            } catch (SQLException e) {
                throw new RuntimeException("Failed to initialize test database", e);
            }
        }
    }
}
```

### Integration Tests

```java
@SpringBootTest
@TestPropertySource(properties = {
    "spring.datasource.url=jdbc:h2:mem:testdb",
    "spring.jpa.hibernate.ddl-auto=none"
})
class EmployeeDAOIntegrationTest {
    
    @Autowired
    private EmployeeDAO employeeDAO;
    
    @Autowired
    private DataSource dataSource;
    
    @Test
    @DisplayName("Should save and retrieve employee successfully")
    void shouldSaveAndRetrieveEmployee() {
        // Given
        Employee employee = Employee.builder()
            .name("John Doe")
            .email("john.doe@example.com")
            .salary(new BigDecimal("75000.00"))
            .departmentId(1)
            .hireDate(LocalDate.now())
            .build();
        
        // When
        Integer savedId = employeeDAO.save(employee);
        Optional<Employee> retrieved = employeeDAO.findById(savedId);
        
        // Then
        assertThat(retrieved).isPresent();
        assertThat(retrieved.get().getName()).isEqualTo("John Doe");
        assertThat(retrieved.get().getEmail()).isEqualTo("john.doe@example.com");
        assertThat(retrieved.get().getSalary()).isEqualByComparingTo(new BigDecimal("75000.00"));
    }
    
    @Test
    @DisplayName("Should handle duplicate email constraint violation")
    void shouldHandleDuplicateEmailConstraint() {
        // Given
        Employee employee1 = createTestEmployee("duplicate@example.com");
        Employee employee2 = createTestEmployee("duplicate@example.com");
        
        // When
        employeeDAO.save(employee1);
        
        // Then
        assertThatThrownBy(() -> employeeDAO.save(employee2))
            .isInstanceOf(DuplicateEntityException.class)
            .hasMessageContaining("already exists");
    }
    
    @Test
    @DisplayName("Should perform batch operations efficiently")
    void shouldPerformBatchOperations() {
        // Given
        List<Employee> employees = IntStream.range(0, 1000)
            .mapToObj(i -> createTestEmployee("user" + i + "@example.com"))
            .collect(Collectors.toList());
        
        // When
        long startTime = System.currentTimeMillis();
        employees.forEach(employeeDAO::save);
        long duration = System.currentTimeMillis() - startTime;
        
        // Then
        long count = employeeDAO.count();
        assertThat(count).isGreaterThanOrEqualTo(1000);
        assertThat(duration).isLessThan(5000); // Should complete within 5 seconds
    }
    
    @Test
    @Transactional
    @Rollback
    @DisplayName("Should rollback transaction on error")
    void shouldRollbackTransactionOnError() {
        // Given
        long initialCount = employeeDAO.count();
        
        // When & Then
        assertThatThrownBy(() -> {
            try (Connection conn = dataSource.getConnection()) {
                conn.setAutoCommit(false);
                
                // This should succeed
                employeeDAO.save(createTestEmployee("valid@example.com"));
                
                // This should fail due to null constraint
                employeeDAO.save(Employee.builder()
                    .name(null) // This will cause constraint violation
                    .email("invalid@example.com")
                    .salary(BigDecimal.TEN)
                    .departmentId(1)
                    .build());
                
                conn.commit();
            }
        }).isInstanceOf(DataAccessException.class);
        
        // Verify rollback
        assertThat(employeeDAO.count()).isEqualTo(initialCount);
    }
    
    private Employee createTestEmployee(String email) {
        return Employee.builder()
            .name("Test User")
            .email(email)
            .salary(new BigDecimal("50000.00"))
            .departmentId(1)
            .hireDate(LocalDate.now())
            .build();
    }
}
```

---

## Security Best Practices

### SQL Injection Prevention

```java
@Service
public class SecureQueryService {
    
    // ✅ SAFE - Using PreparedStatement with parameters
    public List<Employee> searchEmployeesSafe(String namePattern, BigDecimal minSalary) {
        String sql = """
            SELECT * FROM employees 
            WHERE name ILIKE ? 
            AND salary >= ? 
            ORDER BY name
            """;
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            pstmt.setString(1, "%" + namePattern + "%");
            pstmt.setBigDecimal(2, minSalary);
            
            // Process results...
            return processResults(pstmt.executeQuery());
            
        } catch (SQLException e) {
            throw new DataAccessException("Search failed", e);
        }
    }
    
    // ❌ VULNERABLE - Never do this!
    public List<Employee> searchEmployeesUnsafe(String namePattern, BigDecimal minSalary) {
        String sql = "SELECT * FROM employees WHERE name LIKE '%" + namePattern + 
                    "%' AND salary >= " + minSalary + " ORDER BY name";
        
        // This is vulnerable to SQL injection!
        // Example malicious input: namePattern = "'; DROP TABLE employees; --"
        
        try (Connection conn = dataSource.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            
            return processResults(rs);
            
        } catch (SQLException e) {
            throw new DataAccessException("Search failed", e);
        }
    }
    
    // Advanced: Dynamic query building with validation
    public List<Employee> dynamicSearchSecure(SearchCriteria criteria) {
        StringBuilder sql = new StringBuilder("SELECT * FROM employees WHERE 1=1");
        List<Object> parameters = new ArrayList<>();
        
        // Validate and build query safely
        if (criteria.getName() != null) {
            validateInput(criteria.getName(), "name");
            sql.append(" AND name ILIKE ?");
            parameters.add("%" + criteria.getName() + "%");
        }
        
        if (criteria.getDepartment() != null) {
            validateInput(criteria.getDepartment(), "department");
            sql.append(" AND department_id = ?");
            parameters.add(criteria.getDepartment());
        }
        
        if (criteria.getMinSalary() != null) {
            sql.append(" AND salary >= ?");
            parameters.add(criteria.getMinSalary());
        }
        
        // Validate sort column (whitelist approach)
        String sortColumn = validateSortColumn(criteria.getSortBy());
        sql.append(" ORDER BY ").append(sortColumn);
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql.toString())) {
            
            for (int i = 0; i < parameters.size(); i++) {
                pstmt.setObject(i + 1, parameters.get(i));
            }
            
            return processResults(pstmt.executeQuery());
            
        } catch (SQLException e) {
            throw new DataAccessException("Dynamic search failed", e);
        }
    }
    
    private void validateInput(String input, String fieldName) {
        if (input == null || input.trim().isEmpty()) {
            throw new IllegalArgumentException(fieldName + " cannot be empty");
        }
        if (input.length() > 255) {
            throw new IllegalArgumentException(fieldName + " is too long");
        }
        // Check for suspicious patterns
        if (input.contains("'") || input.contains("--") || input.contains(";")) {
            throw new IllegalArgumentException("Invalid characters in " + fieldName);
        }
    }
    
    private String validateSortColumn(String sortBy) {
        // Whitelist approach for sort columns
        Set<String> allowedColumns = Set.of("name", "email", "salary", "hire_date", "department_id");
        
        if (sortBy == null || !allowedColumns.contains(sortBy.toLowerCase())) {
            return "name"; // Default safe column
        }
        
        return sortBy.toLowerCase();
    }
}
```

### Credential Management

```java
@Configuration
public class SecurityConfig {
    
    @Bean
    public DataSource secureDataSource() {
        HikariConfig config = new HikariConfig();
        
        // Use environment variables for credentials
        config.setJdbcUrl(getRequiredEnvVar("DB_URL"));
        config.setUsername(getRequiredEnvVar("DB_USERNAME"));
        config.setPassword(getRequiredEnvVar("DB_PASSWORD"));
        
        // SSL configuration
        config.addDataSourceProperty("ssl", "true");
        config.addDataSourceProperty("sslmode", "require");
        config.addDataSourceProperty("sslcert", getEnvVar("DB_SSL_CERT"));
        config.addDataSourceProperty("sslkey", getEnvVar("DB_SSL_KEY"));
        config.addDataSourceProperty("sslrootcert", getEnvVar("DB_SSL_ROOT_CERT"));
        
        // Connection security
        config.setConnectionTimeout(10000); // 10 seconds max
        config.setValidationTimeout(5000);  // 5 seconds validation
        config.setLeakDetectionThreshold(60000); // Detect leaks after 1 minute
        
        return new HikariDataSource(config);
    }
    
    private String getRequiredEnvVar(String name) {
        String value = System.getenv(name);
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalStateException("Required environment variable not set: " + name);
        }
        return value;
    }
    
    private String getEnvVar(String name) {
        return System.getenv(name);
    }
}
```

This comprehensive guide provides modern, production-ready patterns for working with JDBC in contemporary Java applications. Each section includes practical examples that can be adapted to specific use cases while maintaining security, performance, and maintainability standards.
