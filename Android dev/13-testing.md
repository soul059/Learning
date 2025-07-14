# Testing

## Table of Contents
- [Testing Overview](#testing-overview)
- [Unit Testing](#unit-testing)
- [Instrumentation Testing](#instrumentation-testing)
- [UI Testing with Espresso](#ui-testing-with-espresso)
- [Mocking](#mocking)
- [Test-Driven Development](#test-driven-development)
- [Testing Best Practices](#testing-best-practices)
- [Continuous Integration](#continuous-integration)

## Testing Overview

### Testing Pyramid
1. **Unit Tests** (70%) - Fast, isolated tests for individual components
2. **Integration Tests** (20%) - Test component interactions
3. **UI/End-to-End Tests** (10%) - Test complete user workflows

### Testing Dependencies
```gradle
// app/build.gradle
dependencies {
    // Unit testing
    testImplementation 'junit:junit:4.13.2'
    testImplementation 'org.mockito:mockito-core:5.1.1'
    testImplementation 'org.mockito:mockito-inline:5.1.1'
    testImplementation 'androidx.arch.core:core-testing:2.2.0'
    testImplementation 'org.robolectric:robolectric:4.10.3'
    
    // Android instrumentation testing
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
    androidTestImplementation 'androidx.test.espresso:espresso-intents:3.5.1'
    androidTestImplementation 'androidx.test.espresso:espresso-contrib:3.5.1'
    androidTestImplementation 'androidx.test:runner:1.5.2'
    androidTestImplementation 'androidx.test:rules:1.5.0'
    
    // UI testing
    androidTestImplementation 'androidx.test.uiautomator:uiautomator:2.2.0'
    
    // Testing fragments
    debugImplementation 'androidx.fragment:fragment-testing:1.6.1'
}
```

## Unit Testing

### Basic Unit Test Example
```java
// src/test/java/com/example/CalculatorTest.java
public class CalculatorTest {
    
    private Calculator calculator;
    
    @Before
    public void setUp() {
        calculator = new Calculator();
    }
    
    @Test
    public void addition_isCorrect() {
        // Arrange
        int a = 5;
        int b = 3;
        int expected = 8;
        
        // Act
        int result = calculator.add(a, b);
        
        // Assert
        assertEquals(expected, result);
    }
    
    @Test
    public void division_byZero_throwsException() {
        // Arrange
        int a = 10;
        int b = 0;
        
        // Act & Assert
        assertThrows(ArithmeticException.class, () -> {
            calculator.divide(a, b);
        });
    }
    
    @Test
    public void multiplication_withNegativeNumbers() {
        // Arrange
        int a = -5;
        int b = 3;
        int expected = -15;
        
        // Act
        int result = calculator.multiply(a, b);
        
        // Assert
        assertEquals(expected, result);
    }
}

// Calculator class being tested
public class Calculator {
    
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
    
    public int multiply(int a, int b) {
        return a * b;
    }
    
    public int divide(int a, int b) {
        if (b == 0) {
            throw new ArithmeticException("Division by zero is not allowed");
        }
        return a / b;
    }
}
```

### Testing with LiveData and ViewModel
```java
// UserViewModel being tested
public class UserViewModel extends ViewModel {
    
    private UserRepository repository;
    private MutableLiveData<List<User>> users;
    private MutableLiveData<String> error;
    
    public UserViewModel(UserRepository repository) {
        this.repository = repository;
        this.users = new MutableLiveData<>();
        this.error = new MutableLiveData<>();
    }
    
    public LiveData<List<User>> getUsers() {
        return users;
    }
    
    public LiveData<String> getError() {
        return error;
    }
    
    public void loadUsers() {
        repository.getUsers(new UserRepository.Callback() {
            @Override
            public void onSuccess(List<User> userList) {
                users.setValue(userList);
            }
            
            @Override
            public void onError(String errorMessage) {
                error.setValue(errorMessage);
            }
        });
    }
}

// Unit test for UserViewModel
public class UserViewModelTest {
    
    @Rule
    public InstantTaskExecutorRule instantExecutorRule = new InstantTaskExecutorRule();
    
    @Mock
    private UserRepository mockRepository;
    
    private UserViewModel userViewModel;
    
    @Before
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        userViewModel = new UserViewModel(mockRepository);
    }
    
    @Test
    public void loadUsers_success_updatesLiveData() {
        // Arrange
        List<User> expectedUsers = Arrays.asList(
            new User(1, "John Doe", "john@example.com"),
            new User(2, "Jane Smith", "jane@example.com")
        );
        
        // Mock repository behavior
        doAnswer(invocation -> {
            UserRepository.Callback callback = invocation.getArgument(0);
            callback.onSuccess(expectedUsers);
            return null;
        }).when(mockRepository).getUsers(any(UserRepository.Callback.class));
        
        // Act
        userViewModel.loadUsers();
        
        // Assert
        assertEquals(expectedUsers, userViewModel.getUsers().getValue());
        assertNull(userViewModel.getError().getValue());
    }
    
    @Test
    public void loadUsers_error_updatesErrorLiveData() {
        // Arrange
        String expectedError = "Network error";
        
        // Mock repository behavior
        doAnswer(invocation -> {
            UserRepository.Callback callback = invocation.getArgument(0);
            callback.onError(expectedError);
            return null;
        }).when(mockRepository).getUsers(any(UserRepository.Callback.class));
        
        // Act
        userViewModel.loadUsers();
        
        // Assert
        assertEquals(expectedError, userViewModel.getError().getValue());
        assertNull(userViewModel.getUsers().getValue());
    }
    
    @Test
    public void loadUsers_callsRepository() {
        // Act
        userViewModel.loadUsers();
        
        // Assert
        verify(mockRepository).getUsers(any(UserRepository.Callback.class));
    }
}
```

### Testing Utility Classes
```java
// Utility class being tested
public class StringUtils {
    
    public static boolean isEmpty(String str) {
        return str == null || str.trim().isEmpty();
    }
    
    public static String capitalize(String str) {
        if (isEmpty(str)) {
            return str;
        }
        return str.substring(0, 1).toUpperCase() + str.substring(1).toLowerCase();
    }
    
    public static String reverse(String str) {
        if (isEmpty(str)) {
            return str;
        }
        return new StringBuilder(str).reverse().toString();
    }
    
    public static int countWords(String str) {
        if (isEmpty(str)) {
            return 0;
        }
        return str.trim().split("\\s+").length;
    }
}

// Unit test for StringUtils
public class StringUtilsTest {
    
    @Test
    public void isEmpty_withNull_returnsTrue() {
        assertTrue(StringUtils.isEmpty(null));
    }
    
    @Test
    public void isEmpty_withEmptyString_returnsTrue() {
        assertTrue(StringUtils.isEmpty(""));
    }
    
    @Test
    public void isEmpty_withWhitespace_returnsTrue() {
        assertTrue(StringUtils.isEmpty("   "));
    }
    
    @Test
    public void isEmpty_withText_returnsFalse() {
        assertFalse(StringUtils.isEmpty("Hello"));
    }
    
    @Test
    public void capitalize_withLowercase_returnsCapitalized() {
        assertEquals("Hello", StringUtils.capitalize("hello"));
    }
    
    @Test
    public void capitalize_withUppercase_returnsCapitalized() {
        assertEquals("Hello", StringUtils.capitalize("HELLO"));
    }
    
    @Test
    public void capitalize_withMixedCase_returnsCapitalized() {
        assertEquals("Hello", StringUtils.capitalize("hELLo"));
    }
    
    @Test
    public void capitalize_withEmptyString_returnsEmpty() {
        assertEquals("", StringUtils.capitalize(""));
    }
    
    @Test
    public void capitalize_withNull_returnsNull() {
        assertNull(StringUtils.capitalize(null));
    }
    
    @Test
    public void reverse_withNormalString_returnsReversed() {
        assertEquals("olleH", StringUtils.reverse("Hello"));
    }
    
    @Test
    public void reverse_withEmptyString_returnsEmpty() {
        assertEquals("", StringUtils.reverse(""));
    }
    
    @Test
    public void countWords_withSingleWord_returnsOne() {
        assertEquals(1, StringUtils.countWords("Hello"));
    }
    
    @Test
    public void countWords_withMultipleWords_returnsCorrectCount() {
        assertEquals(3, StringUtils.countWords("Hello World Test"));
    }
    
    @Test
    public void countWords_withExtraSpaces_returnsCorrectCount() {
        assertEquals(3, StringUtils.countWords("  Hello   World   Test  "));
    }
    
    @Test
    public void countWords_withEmptyString_returnsZero() {
        assertEquals(0, StringUtils.countWords(""));
    }
}
```

## Instrumentation Testing

### Basic Instrumentation Test
```java
// src/androidTest/java/com/example/MainActivityTest.java
@RunWith(AndroidJUnit4.class)
public class MainActivityTest {
    
    @Rule
    public ActivityTestRule<MainActivity> activityRule = 
        new ActivityTestRule<>(MainActivity.class);
    
    @Test
    public void useAppContext() {
        // Context of the app under test
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertEquals("com.example.myapp", appContext.getPackageName());
    }
    
    @Test
    public void activity_launches_successfully() {
        // Just check that the activity launches without crashing
        assertNotNull(activityRule.getActivity());
    }
    
    @Test
    public void button_exists_and_isClickable() {
        onView(withId(R.id.button)).check(matches(isDisplayed()));
        onView(withId(R.id.button)).check(matches(isClickable()));
    }
}
```

### Testing Database Operations
```java
// Database test
@RunWith(AndroidJUnit4.class)
public class UserDatabaseTest {
    
    private UserDatabase database;
    private UserDao userDao;
    
    @Before
    public void createDb() {
        Context context = ApplicationProvider.getApplicationContext();
        database = Room.inMemoryDatabaseBuilder(context, UserDatabase.class).build();
        userDao = database.userDao();
    }
    
    @After
    public void closeDb() throws IOException {
        database.close();
    }
    
    @Test
    public void insertAndGetUser() throws Exception {
        // Arrange
        User user = new User(1, "John Doe", "john@example.com");
        
        // Act
        userDao.insert(user);
        List<User> allUsers = userDao.getAllUsers();
        
        // Assert
        assertThat(allUsers.size(), equalTo(1));
        assertThat(allUsers.get(0).getName(), equalTo("John Doe"));
        assertThat(allUsers.get(0).getEmail(), equalTo("john@example.com"));
    }
    
    @Test
    public void updateUser() throws Exception {
        // Arrange
        User user = new User(1, "John Doe", "john@example.com");
        userDao.insert(user);
        
        // Act
        user.setName("John Smith");
        userDao.update(user);
        
        // Assert
        User updatedUser = userDao.getUserById(1);
        assertThat(updatedUser.getName(), equalTo("John Smith"));
    }
    
    @Test
    public void deleteUser() throws Exception {
        // Arrange
        User user = new User(1, "John Doe", "john@example.com");
        userDao.insert(user);
        
        // Act
        userDao.delete(user);
        
        // Assert
        List<User> allUsers = userDao.getAllUsers();
        assertThat(allUsers.size(), equalTo(0));
    }
}
```

## UI Testing with Espresso

### Basic Espresso Tests
```java
@RunWith(AndroidJUnit4.class)
@LargeTest
public class LoginActivityTest {
    
    @Rule
    public ActivityTestRule<LoginActivity> activityRule = 
        new ActivityTestRule<>(LoginActivity.class);
    
    @Test
    public void login_withValidCredentials_navigatesToMainActivity() {
        // Type email
        onView(withId(R.id.editTextEmail))
            .perform(typeText("test@example.com"), closeSoftKeyboard());
        
        // Type password
        onView(withId(R.id.editTextPassword))
            .perform(typeText("password123"), closeSoftKeyboard());
        
        // Click login button
        onView(withId(R.id.buttonLogin))
            .perform(click());
        
        // Check that MainActivity is launched
        intended(hasComponent(MainActivity.class.getName()));
    }
    
    @Test
    public void login_withInvalidEmail_showsError() {
        // Type invalid email
        onView(withId(R.id.editTextEmail))
            .perform(typeText("invalid-email"), closeSoftKeyboard());
        
        // Type password
        onView(withId(R.id.editTextPassword))
            .perform(typeText("password123"), closeSoftKeyboard());
        
        // Click login button
        onView(withId(R.id.buttonLogin))
            .perform(click());
        
        // Check error message
        onView(withText("Please enter a valid email address"))
            .check(matches(isDisplayed()));
    }
    
    @Test
    public void login_withEmptyFields_showsValidationErrors() {
        // Click login without entering anything
        onView(withId(R.id.buttonLogin))
            .perform(click());
        
        // Check validation errors
        onView(withText("Email is required"))
            .check(matches(isDisplayed()));
        onView(withText("Password is required"))
            .check(matches(isDisplayed()));
    }
}
```

### Testing RecyclerView
```java
@RunWith(AndroidJUnit4.class)
public class UserListActivityTest {
    
    @Rule
    public ActivityTestRule<UserListActivity> activityRule = 
        new ActivityTestRule<>(UserListActivity.class);
    
    @Test
    public void userList_displaysUsers() {
        // Check that RecyclerView is displayed
        onView(withId(R.id.recyclerViewUsers))
            .check(matches(isDisplayed()));
        
        // Check that first user is displayed
        onView(withRecyclerView(R.id.recyclerViewUsers).atPosition(0))
            .check(matches(hasDescendant(withText("John Doe"))));
    }
    
    @Test
    public void userList_clickItem_opensUserDetail() {
        // Click on first user
        onView(withRecyclerView(R.id.recyclerViewUsers).atPosition(0))
            .perform(click());
        
        // Check that UserDetailActivity is opened
        intended(hasComponent(UserDetailActivity.class.getName()));
    }
    
    @Test
    public void userList_swipeToRefresh_refreshesList() {
        // Perform swipe to refresh
        onView(withId(R.id.swipeRefreshLayout))
            .perform(swipeDown());
        
        // Check that progress indicator is shown
        onView(withId(R.id.swipeRefreshLayout))
            .check(matches(isRefreshing()));
    }
    
    // Custom matcher for RecyclerView
    public static RecyclerViewMatcher withRecyclerView(final int recyclerViewId) {
        return new RecyclerViewMatcher(recyclerViewId);
    }
}

// Custom RecyclerView matcher
public class RecyclerViewMatcher {
    private final int recyclerViewId;
    
    public RecyclerViewMatcher(int recyclerViewId) {
        this.recyclerViewId = recyclerViewId;
    }
    
    public Matcher<View> atPosition(final int position) {
        return atPositionOnView(position, -1);
    }
    
    public Matcher<View> atPositionOnView(final int position, final int targetViewId) {
        return new TypeSafeMatcher<View>() {
            Resources resources = null;
            View childView;
            
            public void describeTo(Description description) {
                String idDescription = Integer.toString(recyclerViewId);
                if (this.resources != null) {
                    try {
                        idDescription = this.resources.getResourceName(recyclerViewId);
                    } catch (Resources.NotFoundException var4) {
                        idDescription = String.format("%s (resource name not found)", recyclerViewId);
                    }
                }
                description.appendText("RecyclerView with id: " + idDescription + " at position: " + position);
            }
            
            public boolean matchesSafely(View view) {
                this.resources = view.getResources();
                
                if (childView == null) {
                    RecyclerView recyclerView = view.getRootView().findViewById(recyclerViewId);
                    if (recyclerView != null && recyclerView.getId() == recyclerViewId) {
                        RecyclerView.ViewHolder viewHolder = recyclerView.findViewHolderForAdapterPosition(position);
                        if (viewHolder != null) {
                            childView = viewHolder.itemView;
                        }
                    } else {
                        return false;
                    }
                }
                
                if (targetViewId == -1) {
                    return view == childView;
                } else {
                    View targetView = childView.findViewById(targetViewId);
                    return view == targetView;
                }
            }
        };
    }
}
```

### Testing Navigation
```java
@RunWith(AndroidJUnit4.class)
public class NavigationTest {
    
    @Rule
    public ActivityTestRule<MainActivity> activityRule = 
        new ActivityTestRule<>(MainActivity.class);
    
    @Test
    public void bottomNavigation_clickAllTabs_navigatesCorrectly() {
        // Click Home tab
        onView(withId(R.id.navigation_home))
            .perform(click());
        onView(withId(R.id.fragment_home))
            .check(matches(isDisplayed()));
        
        // Click Search tab
        onView(withId(R.id.navigation_search))
            .perform(click());
        onView(withId(R.id.fragment_search))
            .check(matches(isDisplayed()));
        
        // Click Profile tab
        onView(withId(R.id.navigation_profile))
            .perform(click());
        onView(withId(R.id.fragment_profile))
            .check(matches(isDisplayed()));
    }
    
    @Test
    public void navigationDrawer_opensAndCloses() {
        // Open navigation drawer
        onView(withId(R.id.drawer_layout))
            .perform(open());
        
        // Check that navigation view is displayed
        onView(withId(R.id.nav_view))
            .check(matches(isDisplayed()));
        
        // Click a menu item
        onView(withText("Settings"))
            .perform(click());
        
        // Check that drawer is closed
        onView(withId(R.id.drawer_layout))
            .check(matches(isClosed()));
    }
}
```

## Mocking

### Mockito Examples
```java
public class UserServiceTest {
    
    @Mock
    private UserRepository mockRepository;
    
    @Mock
    private NetworkManager mockNetworkManager;
    
    @InjectMocks
    private UserService userService;
    
    @Before
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }
    
    @Test
    public void getUserById_success_returnsUser() {
        // Arrange
        int userId = 1;
        User expectedUser = new User(userId, "John Doe", "john@example.com");
        when(mockRepository.getUserById(userId)).thenReturn(expectedUser);
        
        // Act
        User result = userService.getUserById(userId);
        
        // Assert
        assertEquals(expectedUser, result);
        verify(mockRepository).getUserById(userId);
    }
    
    @Test
    public void createUser_success_callsRepository() {
        // Arrange
        User newUser = new User(0, "Jane Doe", "jane@example.com");
        User savedUser = new User(1, "Jane Doe", "jane@example.com");
        when(mockRepository.createUser(newUser)).thenReturn(savedUser);
        
        // Act
        User result = userService.createUser(newUser);
        
        // Assert
        assertEquals(savedUser, result);
        verify(mockRepository).createUser(newUser);
    }
    
    @Test
    public void syncUsers_networkAvailable_syncsWithServer() {
        // Arrange
        when(mockNetworkManager.isNetworkAvailable()).thenReturn(true);
        List<User> serverUsers = Arrays.asList(
            new User(1, "John Doe", "john@example.com"),
            new User(2, "Jane Smith", "jane@example.com")
        );
        when(mockNetworkManager.fetchUsersFromServer()).thenReturn(serverUsers);
        
        // Act
        userService.syncUsers();
        
        // Assert
        verify(mockNetworkManager).isNetworkAvailable();
        verify(mockNetworkManager).fetchUsersFromServer();
        verify(mockRepository).saveUsers(serverUsers);
    }
    
    @Test
    public void syncUsers_networkUnavailable_doesNotSync() {
        // Arrange
        when(mockNetworkManager.isNetworkAvailable()).thenReturn(false);
        
        // Act
        userService.syncUsers();
        
        // Assert
        verify(mockNetworkManager).isNetworkAvailable();
        verify(mockNetworkManager, never()).fetchUsersFromServer();
        verify(mockRepository, never()).saveUsers(any());
    }
    
    @Test
    public void deleteUser_callsRepositoryAndNotifiesListeners() {
        // Arrange
        int userId = 1;
        UserService.OnUserDeletedListener mockListener = mock(UserService.OnUserDeletedListener.class);
        userService.addOnUserDeletedListener(mockListener);
        
        // Act
        userService.deleteUser(userId);
        
        // Assert
        verify(mockRepository).deleteUser(userId);
        verify(mockListener).onUserDeleted(userId);
    }
}
```

### PowerMockito for Static Methods
```java
@RunWith(PowerMockRunner.class)
@PrepareForTest({TextUtils.class, Log.class})
public class StaticMethodTest {
    
    @Test
    public void testStaticMethod() {
        // Mock static method
        PowerMockito.mockStatic(TextUtils.class);
        when(TextUtils.isEmpty(anyString())).thenReturn(false);
        
        // Test code that uses TextUtils.isEmpty()
        boolean result = MyClass.isValidString("test");
        
        // Verify
        assertTrue(result);
        PowerMockito.verifyStatic(TextUtils.class);
        TextUtils.isEmpty("test");
    }
}
```

## Test-Driven Development

### TDD Example - Calculator Feature
```java
// 1. Write failing test first
public class CalculatorTDDTest {
    
    private Calculator calculator;
    
    @Before
    public void setUp() {
        calculator = new Calculator();
    }
    
    @Test
    public void calculate_simpleAddition_returnsCorrectResult() {
        // This test will fail initially because Calculator doesn't exist
        String expression = "2 + 3";
        double expected = 5.0;
        
        double result = calculator.calculate(expression);
        
        assertEquals(expected, result, 0.001);
    }
    
    @Test
    public void calculate_simpleSubtraction_returnsCorrectResult() {
        String expression = "5 - 2";
        double expected = 3.0;
        
        double result = calculator.calculate(expression);
        
        assertEquals(expected, result, 0.001);
    }
    
    @Test
    public void calculate_multiplication_returnsCorrectResult() {
        String expression = "4 * 3";
        double expected = 12.0;
        
        double result = calculator.calculate(expression);
        
        assertEquals(expected, result, 0.001);
    }
    
    @Test
    public void calculate_division_returnsCorrectResult() {
        String expression = "8 / 2";
        double expected = 4.0;
        
        double result = calculator.calculate(expression);
        
        assertEquals(expected, result, 0.001);
    }
    
    @Test
    public void calculate_complexExpression_returnsCorrectResult() {
        String expression = "2 + 3 * 4";
        double expected = 14.0; // Order of operations: 3*4=12, 2+12=14
        
        double result = calculator.calculate(expression);
        
        assertEquals(expected, result, 0.001);
    }
    
    @Test
    public void calculate_divisionByZero_throwsException() {
        String expression = "5 / 0";
        
        assertThrows(ArithmeticException.class, () -> {
            calculator.calculate(expression);
        });
    }
    
    @Test
    public void calculate_invalidExpression_throwsException() {
        String expression = "2 + + 3";
        
        assertThrows(IllegalArgumentException.class, () -> {
            calculator.calculate(expression);
        });
    }
}

// 2. Implement minimum code to make tests pass
public class Calculator {
    
    public double calculate(String expression) {
        if (expression == null || expression.trim().isEmpty()) {
            throw new IllegalArgumentException("Expression cannot be null or empty");
        }
        
        // Remove spaces
        expression = expression.replaceAll("\\s+", "");
        
        // Validate expression
        if (!isValidExpression(expression)) {
            throw new IllegalArgumentException("Invalid expression: " + expression);
        }
        
        // Parse and evaluate
        return evaluateExpression(expression);
    }
    
    private boolean isValidExpression(String expression) {
        // Basic validation - contains only numbers, operators, and dots
        return expression.matches("^[0-9+\\-*/().]+$") && 
               !expression.matches(".*[+\\-*/]{2,}.*"); // No consecutive operators
    }
    
    private double evaluateExpression(String expression) {
        // Simple recursive descent parser
        return parseExpression(new ExpressionParser(expression));
    }
    
    private double parseExpression(ExpressionParser parser) {
        double result = parseTerm(parser);
        
        while (parser.hasNext()) {
            char operator = parser.peek();
            if (operator == '+' || operator == '-') {
                parser.next(); // consume operator
                double operand = parseTerm(parser);
                if (operator == '+') {
                    result += operand;
                } else {
                    result -= operand;
                }
            } else {
                break;
            }
        }
        
        return result;
    }
    
    private double parseTerm(ExpressionParser parser) {
        double result = parseFactor(parser);
        
        while (parser.hasNext()) {
            char operator = parser.peek();
            if (operator == '*' || operator == '/') {
                parser.next(); // consume operator
                double operand = parseFactor(parser);
                if (operator == '*') {
                    result *= operand;
                } else {
                    if (operand == 0) {
                        throw new ArithmeticException("Division by zero");
                    }
                    result /= operand;
                }
            } else {
                break;
            }
        }
        
        return result;
    }
    
    private double parseFactor(ExpressionParser parser) {
        if (parser.peek() == '(') {
            parser.next(); // consume '('
            double result = parseExpression(parser);
            if (parser.peek() != ')') {
                throw new IllegalArgumentException("Missing closing parenthesis");
            }
            parser.next(); // consume ')'
            return result;
        }
        
        return parseNumber(parser);
    }
    
    private double parseNumber(ExpressionParser parser) {
        StringBuilder number = new StringBuilder();
        
        while (parser.hasNext() && (Character.isDigit(parser.peek()) || parser.peek() == '.')) {
            number.append(parser.next());
        }
        
        if (number.length() == 0) {
            throw new IllegalArgumentException("Expected number");
        }
        
        return Double.parseDouble(number.toString());
    }
    
    private static class ExpressionParser {
        private final String expression;
        private int position = 0;
        
        public ExpressionParser(String expression) {
            this.expression = expression;
        }
        
        public boolean hasNext() {
            return position < expression.length();
        }
        
        public char peek() {
            if (!hasNext()) {
                throw new IllegalArgumentException("Unexpected end of expression");
            }
            return expression.charAt(position);
        }
        
        public char next() {
            if (!hasNext()) {
                throw new IllegalArgumentException("Unexpected end of expression");
            }
            return expression.charAt(position++);
        }
    }
}

// 3. Refactor and improve code while keeping tests green
```

## Testing Best Practices

### Test Organization
```java
public class UserManagerTest {
    
    // Test fixtures
    private UserManager userManager;
    private User validUser;
    private User invalidUser;
    
    @Before
    public void setUp() {
        userManager = new UserManager();
        validUser = new User(1, "John Doe", "john@example.com");
        invalidUser = new User(0, "", "invalid-email");
    }
    
    @After
    public void tearDown() {
        // Clean up resources if needed
    }
    
    // Group related tests using nested classes
    @RunWith(Suite.class)
    @Suite.SuiteClasses({
        UserValidationTests.class,
        UserCrudOperationTests.class,
        UserBusinessLogicTests.class
    })
    public static class UserManagerTestSuite {
    }
    
    public static class UserValidationTests {
        
        @Test
        public void validateUser_withValidData_returnsTrue() {
            // Test implementation
        }
        
        @Test
        public void validateUser_withInvalidEmail_returnsFalse() {
            // Test implementation
        }
        
        @Test
        public void validateUser_withEmptyName_returnsFalse() {
            // Test implementation
        }
    }
    
    public static class UserCrudOperationTests {
        
        @Test
        public void createUser_withValidData_succeeds() {
            // Test implementation
        }
        
        @Test
        public void updateUser_withValidData_succeeds() {
            // Test implementation
        }
        
        @Test
        public void deleteUser_withValidId_succeeds() {
            // Test implementation
        }
    }
}
```

### Parameterized Tests
```java
@RunWith(Parameterized.class)
public class EmailValidationTest {
    
    private String email;
    private boolean expected;
    
    public EmailValidationTest(String email, boolean expected) {
        this.email = email;
        this.expected = expected;
    }
    
    @Parameterized.Parameters(name = "{index}: isValidEmail({0}) = {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {"test@example.com", true},
            {"user.name@domain.co.uk", true},
            {"user+tag@example.com", true},
            {"invalid-email", false},
            {"@example.com", false},
            {"user@", false},
            {"", false},
            {null, false}
        });
    }
    
    @Test
    public void testEmailValidation() {
        assertEquals(expected, EmailValidator.isValid(email));
    }
}
```

### Test Data Builders
```java
public class UserTestDataBuilder {
    
    private int id = 1;
    private String name = "John Doe";
    private String email = "john@example.com";
    private int age = 25;
    private String phone = "123-456-7890";
    
    public UserTestDataBuilder withId(int id) {
        this.id = id;
        return this;
    }
    
    public UserTestDataBuilder withName(String name) {
        this.name = name;
        return this;
    }
    
    public UserTestDataBuilder withEmail(String email) {
        this.email = email;
        return this;
    }
    
    public UserTestDataBuilder withAge(int age) {
        this.age = age;
        return this;
    }
    
    public UserTestDataBuilder withPhone(String phone) {
        this.phone = phone;
        return this;
    }
    
    public User build() {
        return new User(id, name, email, age, phone);
    }
    
    // Convenience methods for common test scenarios
    public static UserTestDataBuilder aValidUser() {
        return new UserTestDataBuilder();
    }
    
    public static UserTestDataBuilder anInvalidUser() {
        return new UserTestDataBuilder()
            .withName("")
            .withEmail("invalid-email");
    }
    
    public static UserTestDataBuilder aMinorUser() {
        return new UserTestDataBuilder()
            .withAge(16);
    }
}

// Usage in tests
@Test
public void createUser_withValidData_succeeds() {
    User user = UserTestDataBuilder.aValidUser()
        .withName("Jane Doe")
        .withEmail("jane@example.com")
        .build();
    
    boolean result = userManager.createUser(user);
    
    assertTrue(result);
}
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/android.yml
name: Android CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up JDK 11
      uses: actions/setup-java@v3
      with:
        java-version: '11'
        distribution: 'temurin'
        
    - name: Cache Gradle packages
      uses: actions/cache@v3
      with:
        path: |
          ~/.gradle/caches
          ~/.gradle/wrapper
        key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}
        restore-keys: |
          ${{ runner.os }}-gradle-
          
    - name: Grant execute permission for gradlew
      run: chmod +x gradlew
      
    - name: Run unit tests
      run: ./gradlew test
      
    - name: Run connected tests
      uses: reactivecircus/android-emulator-runner@v2
      with:
        api-level: 29
        script: ./gradlew connectedAndroidTest
        
    - name: Generate test report
      run: ./gradlew jacocoTestReport
      
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      
    - name: Build APK
      run: ./gradlew assembleDebug
      
    - name: Upload APK
      uses: actions/upload-artifact@v3
      with:
        name: app-debug
        path: app/build/outputs/apk/debug/app-debug.apk
```

### Test Coverage Configuration
```gradle
// app/build.gradle
apply plugin: 'jacoco'

jacoco {
    toolVersion = "0.8.8"
}

android {
    buildTypes {
        debug {
            testCoverageEnabled true
        }
    }
}

task jacocoTestReport(type: JacocoReport, dependsOn: ['testDebugUnitTest', 'createDebugCoverageReport']) {
    reports {
        xml.enabled = true
        html.enabled = true
    }

    def fileFilter = [
        '**/R.class',
        '**/R$*.class',
        '**/BuildConfig.*',
        '**/Manifest*.*',
        '**/*Test*.*',
        'android/**/*.*'
    ]

    def debugTree = fileTree(dir: "$project.buildDir/intermediates/javac/debug", excludes: fileFilter)
    def mainSrc = "$project.projectDir/src/main/java"

    sourceDirectories.setFrom(files([mainSrc]))
    classDirectories.setFrom(files([debugTree]))
    executionData.setFrom(fileTree(dir: project.buildDir, includes: [
        'jacoco/testDebugUnitTest.exec', 'outputs/code_coverage/debugAndroidTest/connected/**/*.ec'
    ]))
}
```

Testing is essential for maintaining code quality and ensuring your Android app works correctly. Implement a comprehensive testing strategy that includes unit tests, integration tests, and UI tests to catch bugs early and maintain confidence in your codebase.
