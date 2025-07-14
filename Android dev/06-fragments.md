# Fragments

## Table of Contents
- [What are Fragments](#what-are-fragments)
- [Fragment Lifecycle](#fragment-lifecycle)
- [Creating Fragments](#creating-fragments)
- [Fragment Manager](#fragment-manager)
- [Fragment Transactions](#fragment-transactions)
- [Communication](#communication)
- [Fragment Types](#fragment-types)
- [Best Practices](#best-practices)

## What are Fragments

A **Fragment** represents a reusable portion of your app's UI. Fragments define and manage their own layout, have their own lifecycle, and can handle their own input events.

### Key Benefits
- **Modularity**: Reusable UI components
- **Flexibility**: Adapt to different screen sizes
- **Navigation**: Support for complex navigation patterns
- **Memory Efficiency**: Better resource management

### Fragment vs Activity
| Fragment | Activity |
|----------|----------|
| Part of an activity | Standalone component |
| Has its own lifecycle | Has its own lifecycle |
| Cannot exist alone | Can exist independently |
| Lightweight | Heavier resource usage |
| Better for tablets | Better for phones |

## Fragment Lifecycle

### Lifecycle Methods
```
onAttach() → onCreate() → onCreateView() → onViewCreated() → onStart() → onResume()
    ↑                                                                           ↓
onDetach() ← onDestroy() ← onDestroyView() ← onStop() ← onPause() ←─────────────┘
```

### Lifecycle in Detail
```java
public class ExampleFragment extends Fragment {
    
    @Override
    public void onAttach(@NonNull Context context) {
        super.onAttach(context);
        // Fragment is attached to activity
        Log.d(TAG, "onAttach");
    }
    
    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Fragment is created, initialize non-UI components
        Log.d(TAG, "onCreate");
    }
    
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, 
                           @Nullable ViewGroup container, 
                           @Nullable Bundle savedInstanceState) {
        // Create and return the fragment's UI
        Log.d(TAG, "onCreateView");
        return inflater.inflate(R.layout.fragment_example, container, false);
    }
    
    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        // UI is created, initialize UI components
        Log.d(TAG, "onViewCreated");
        initializeViews(view);
    }
    
    @Override
    public void onStart() {
        super.onStart();
        // Fragment becomes visible
        Log.d(TAG, "onStart");
    }
    
    @Override
    public void onResume() {
        super.onResume();
        // Fragment is active and interacting with user
        Log.d(TAG, "onResume");
    }
    
    @Override
    public void onPause() {
        super.onPause();
        // Fragment loses focus
        Log.d(TAG, "onPause");
    }
    
    @Override
    public void onStop() {
        super.onStop();
        // Fragment is no longer visible
        Log.d(TAG, "onStop");
    }
    
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // Fragment's view is destroyed
        Log.d(TAG, "onDestroyView");
    }
    
    @Override
    public void onDestroy() {
        super.onDestroy();
        // Fragment is destroyed
        Log.d(TAG, "onDestroy");
    }
    
    @Override
    public void onDetach() {
        super.onDetach();
        // Fragment is detached from activity
        Log.d(TAG, "onDetach");
    }
}
```

## Creating Fragments

### 1. Basic Fragment
```java
public class HomeFragment extends Fragment {
    
    private TextView titleText;
    private Button actionButton;
    
    public HomeFragment() {
        // Required empty public constructor
    }
    
    public static HomeFragment newInstance(String title) {
        HomeFragment fragment = new HomeFragment();
        Bundle args = new Bundle();
        args.putString("title", title);
        fragment.setArguments(args);
        return fragment;
    }
    
    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // Get arguments
        if (getArguments() != null) {
            String title = getArguments().getString("title");
        }
    }
    
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, 
                           @Nullable ViewGroup container, 
                           @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_home, container, false);
    }
    
    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        titleText = view.findViewById(R.id.titleText);
        actionButton = view.findViewById(R.id.actionButton);
        
        actionButton.setOnClickListener(v -> {
            // Handle button click
            onActionButtonClicked();
        });
    }
    
    private void onActionButtonClicked() {
        // Communicate with activity or other fragments
        if (getActivity() instanceof MainActivity) {
            ((MainActivity) getActivity()).onFragmentAction("Home action performed");
        }
    }
}
```

### 2. Fragment Layout
```xml
<!-- res/layout/fragment_home.xml -->
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <TextView
        android:id="@+id/titleText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Home Fragment"
        android:textSize="24sp"
        android:textStyle="bold"
        android:gravity="center" />

    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="This is the home fragment content"
        android:layout_marginTop="16dp"
        android:gravity="center" />

    <Button
        android:id="@+id/actionButton"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Perform Action"
        android:layout_marginTop="24dp" />

</LinearLayout>
```

### 3. Fragment Container in Activity
```xml
<!-- res/layout/activity_main.xml -->
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical">

    <com.google.android.material.tabs.TabLayout
        android:id="@+id/tabLayout"
        android:layout_width="match_parent"
        android:layout_height="wrap_content" />

    <FrameLayout
        android:id="@+id/fragmentContainer"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1" />

</LinearLayout>
```

## Fragment Manager

### Adding Fragments Programmatically
```java
public class MainActivity extends AppCompatActivity {
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Add fragment if not already added
        if (savedInstanceState == null) {
            HomeFragment homeFragment = HomeFragment.newInstance("Welcome");
            
            getSupportFragmentManager()
                .beginTransaction()
                .add(R.id.fragmentContainer, homeFragment, "HOME_FRAGMENT")
                .commit();
        }
        
        setupTabNavigation();
    }
    
    private void setupTabNavigation() {
        TabLayout tabLayout = findViewById(R.id.tabLayout);
        tabLayout.addTab(tabLayout.newTab().setText("Home"));
        tabLayout.addTab(tabLayout.newTab().setText("Profile"));
        tabLayout.addTab(tabLayout.newTab().setText("Settings"));
        
        tabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                Fragment fragment = null;
                String tag = null;
                
                switch (tab.getPosition()) {
                    case 0:
                        fragment = HomeFragment.newInstance("Home");
                        tag = "HOME_FRAGMENT";
                        break;
                    case 1:
                        fragment = ProfileFragment.newInstance();
                        tag = "PROFILE_FRAGMENT";
                        break;
                    case 2:
                        fragment = SettingsFragment.newInstance();
                        tag = "SETTINGS_FRAGMENT";
                        break;
                }
                
                if (fragment != null) {
                    getSupportFragmentManager()
                        .beginTransaction()
                        .replace(R.id.fragmentContainer, fragment, tag)
                        .commit();
                }
            }
            
            @Override
            public void onTabUnselected(TabLayout.Tab tab) {}
            
            @Override
            public void onTabReselected(TabLayout.Tab tab) {}
        });
    }
}
```

### Finding Fragments
```java
// Find fragment by tag
Fragment fragment = getSupportFragmentManager().findFragmentByTag("HOME_FRAGMENT");

// Find fragment by ID
Fragment fragment = getSupportFragmentManager().findFragmentById(R.id.fragmentContainer);

// Check if fragment exists
if (fragment != null && fragment instanceof HomeFragment) {
    HomeFragment homeFragment = (HomeFragment) fragment;
    // Use the fragment
}
```

## Fragment Transactions

### Basic Operations
```java
FragmentManager fragmentManager = getSupportFragmentManager();
FragmentTransaction transaction = fragmentManager.beginTransaction();

// Add fragment
transaction.add(R.id.container, new HomeFragment(), "HOME");

// Replace fragment
transaction.replace(R.id.container, new ProfileFragment(), "PROFILE");

// Remove fragment
Fragment fragment = fragmentManager.findFragmentByTag("HOME");
if (fragment != null) {
    transaction.remove(fragment);
}

// Hide/show fragments
transaction.hide(fragment);
transaction.show(fragment);

// Commit the transaction
transaction.commit();
```

### Back Stack Management
```java
// Add to back stack
getSupportFragmentManager()
    .beginTransaction()
    .replace(R.id.fragmentContainer, new DetailFragment())
    .addToBackStack("detail")  // Add to back stack
    .commit();

// Pop back stack
getSupportFragmentManager().popBackStack();

// Pop to specific entry
getSupportFragmentManager().popBackStack("detail", FragmentManager.POP_BACK_STACK_INCLUSIVE);

// Clear back stack
getSupportFragmentManager().popBackStack(null, FragmentManager.POP_BACK_STACK_INCLUSIVE);
```

### Transaction Animations
```java
getSupportFragmentManager()
    .beginTransaction()
    .setCustomAnimations(
        R.anim.slide_in_right,  // Enter animation
        R.anim.slide_out_left,  // Exit animation
        R.anim.slide_in_left,   // Pop enter animation
        R.anim.slide_out_right  // Pop exit animation
    )
    .replace(R.id.fragmentContainer, new ProfileFragment())
    .addToBackStack(null)
    .commit();
```

### Animation Resources
```xml
<!-- res/anim/slide_in_right.xml -->
<?xml version="1.0" encoding="utf-8"?>
<set xmlns:android="http://schemas.android.com/apk/res/android">
    <translate
        android:fromXDelta="100%"
        android:toXDelta="0%"
        android:duration="300" />
</set>

<!-- res/anim/slide_out_left.xml -->
<?xml version="1.0" encoding="utf-8"?>
<set xmlns:android="http://schemas.android.com/apk/res/android">
    <translate
        android:fromXDelta="0%"
        android:toXDelta="-100%"
        android:duration="300" />
</set>
```

## Communication

### 1. Fragment to Activity Communication
```java
// Interface in Fragment
public interface OnFragmentInteractionListener {
    void onFragmentInteraction(String data);
    void onFragmentAction(String action);
}

// In Fragment
public class HomeFragment extends Fragment {
    private OnFragmentInteractionListener listener;
    
    @Override
    public void onAttach(@NonNull Context context) {
        super.onAttach(context);
        if (context instanceof OnFragmentInteractionListener) {
            listener = (OnFragmentInteractionListener) context;
        } else {
            throw new RuntimeException(context.toString() 
                + " must implement OnFragmentInteractionListener");
        }
    }
    
    @Override
    public void onDetach() {
        super.onDetach();
        listener = null;
    }
    
    private void sendDataToActivity(String data) {
        if (listener != null) {
            listener.onFragmentInteraction(data);
        }
    }
}

// In Activity
public class MainActivity extends AppCompatActivity 
        implements HomeFragment.OnFragmentInteractionListener {
    
    @Override
    public void onFragmentInteraction(String data) {
        // Handle data from fragment
        Toast.makeText(this, "Received: " + data, Toast.LENGTH_SHORT).show();
    }
    
    @Override
    public void onFragmentAction(String action) {
        // Handle action from fragment
        switch (action) {
            case "navigate_to_profile":
                navigateToProfile();
                break;
        }
    }
}
```

### 2. Activity to Fragment Communication
```java
// In Activity
public void sendDataToFragment(String data) {
    HomeFragment fragment = (HomeFragment) getSupportFragmentManager()
        .findFragmentByTag("HOME_FRAGMENT");
    
    if (fragment != null) {
        fragment.updateData(data);
    }
}

// In Fragment
public void updateData(String data) {
    if (titleText != null) {
        titleText.setText(data);
    }
}
```

### 3. Fragment to Fragment Communication (via Activity)
```java
// Fragment A
private void sendDataToFragmentB(String data) {
    if (getActivity() instanceof MainActivity) {
        ((MainActivity) getActivity()).forwardDataToFragmentB(data);
    }
}

// Activity
public void forwardDataToFragmentB(String data) {
    ProfileFragment fragmentB = (ProfileFragment) getSupportFragmentManager()
        .findFragmentByTag("PROFILE_FRAGMENT");
    
    if (fragmentB != null) {
        fragmentB.receiveData(data);
    }
}

// Fragment B
public void receiveData(String data) {
    // Handle received data
}
```

### 4. Using ViewModel for Communication
```java
// Shared ViewModel
public class SharedViewModel extends ViewModel {
    private MutableLiveData<String> selectedData = new MutableLiveData<>();
    
    public void selectData(String data) {
        selectedData.setValue(data);
    }
    
    public LiveData<String> getSelectedData() {
        return selectedData;
    }
}

// In Fragment A
public class FragmentA extends Fragment {
    private SharedViewModel viewModel;
    
    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        viewModel = new ViewModelProvider(requireActivity()).get(SharedViewModel.class);
    }
    
    private void selectItem(String item) {
        viewModel.selectData(item);
    }
}

// In Fragment B
public class FragmentB extends Fragment {
    private SharedViewModel viewModel;
    
    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        viewModel = new ViewModelProvider(requireActivity()).get(SharedViewModel.class);
        
        viewModel.getSelectedData().observe(this, data -> {
            // Handle data change
            updateUI(data);
        });
    }
}
```

## Fragment Types

### 1. Static Fragments (in XML)
```xml
<!-- res/layout/activity_tablet.xml -->
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="horizontal">

    <fragment
        android:id="@+id/listFragment"
        android:name="com.example.ListFragment"
        android:layout_width="0dp"
        android:layout_height="match_parent"
        android:layout_weight="1" />

    <fragment
        android:id="@+id/detailFragment"
        android:name="com.example.DetailFragment"
        android:layout_width="0dp"
        android:layout_height="match_parent"
        android:layout_weight="2" />

</LinearLayout>
```

### 2. DialogFragment
```java
public class CustomDialogFragment extends DialogFragment {
    
    private OnDialogResultListener listener;
    
    public interface OnDialogResultListener {
        void onDialogPositiveClick(String result);
        void onDialogNegativeClick();
    }
    
    public static CustomDialogFragment newInstance(String title, String message) {
        CustomDialogFragment fragment = new CustomDialogFragment();
        Bundle args = new Bundle();
        args.putString("title", title);
        args.putString("message", message);
        fragment.setArguments(args);
        return fragment;
    }
    
    @NonNull
    @Override
    public Dialog onCreateDialog(@Nullable Bundle savedInstanceState) {
        Bundle args = getArguments();
        String title = args != null ? args.getString("title", "") : "";
        String message = args != null ? args.getString("message", "") : "";
        
        return new AlertDialog.Builder(requireContext())
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("OK", (dialog, which) -> {
                if (listener != null) {
                    listener.onDialogPositiveClick("User clicked OK");
                }
            })
            .setNegativeButton("Cancel", (dialog, which) -> {
                if (listener != null) {
                    listener.onDialogNegativeClick();
                }
            })
            .create();
    }
    
    @Override
    public void onAttach(@NonNull Context context) {
        super.onAttach(context);
        if (context instanceof OnDialogResultListener) {
            listener = (OnDialogResultListener) context;
        }
    }
}

// Usage in Activity
public void showCustomDialog() {
    CustomDialogFragment dialog = CustomDialogFragment.newInstance(
        "Confirmation", "Are you sure you want to continue?");
    dialog.show(getSupportFragmentManager(), "custom_dialog");
}
```

### 3. Bottom Sheet Fragment
```java
public class BottomSheetFragment extends BottomSheetDialogFragment {
    
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, 
                           @Nullable ViewGroup container, 
                           @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_bottom_sheet, container, false);
    }
    
    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        // Setup bottom sheet behavior
        BottomSheetBehavior<View> behavior = BottomSheetBehavior.from((View) view.getParent());
        behavior.setPeekHeight(400);
        behavior.setState(BottomSheetBehavior.STATE_COLLAPSED);
    }
}
```

### 4. ViewPager Fragment
```java
public class ViewPagerFragment extends Fragment {
    
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, 
                           @Nullable ViewGroup container, 
                           @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_viewpager, container, false);
    }
    
    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        ViewPager2 viewPager = view.findViewById(R.id.viewPager);
        TabLayout tabLayout = view.findViewById(R.id.tabLayout);
        
        ViewPagerAdapter adapter = new ViewPagerAdapter(this);
        viewPager.setAdapter(adapter);
        
        new TabLayoutMediator(tabLayout, viewPager, (tab, position) -> {
            tab.setText("Tab " + (position + 1));
        }).attach();
    }
    
    private static class ViewPagerAdapter extends FragmentStateAdapter {
        
        public ViewPagerAdapter(@NonNull Fragment fragment) {
            super(fragment);
        }
        
        @NonNull
        @Override
        public Fragment createFragment(int position) {
            switch (position) {
                case 0:
                    return new PageOneFragment();
                case 1:
                    return new PageTwoFragment();
                default:
                    return new PageThreeFragment();
            }
        }
        
        @Override
        public int getItemCount() {
            return 3;
        }
    }
}
```

## Best Practices

### 1. Use Static Factory Methods
```java
public class UserFragment extends Fragment {
    private static final String ARG_USER_ID = "user_id";
    private static final String ARG_USER_NAME = "user_name";
    
    // Recommended factory method
    public static UserFragment newInstance(String userId, String userName) {
        UserFragment fragment = new UserFragment();
        Bundle args = new Bundle();
        args.putString(ARG_USER_ID, userId);
        args.putString(ARG_USER_NAME, userName);
        fragment.setArguments(args);
        return fragment;
    }
    
    // Don't create custom constructors with parameters
    public UserFragment() {
        // Required empty public constructor
    }
}
```

### 2. Handle Configuration Changes
```java
@Override
public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
    super.onViewCreated(view, savedInstanceState);
    
    if (savedInstanceState != null) {
        // Restore state
        String savedText = savedInstanceState.getString("saved_text");
        editText.setText(savedText);
    }
}

@Override
public void onSaveInstanceState(@NonNull Bundle outState) {
    super.onSaveInstanceState(outState);
    
    // Save state
    if (editText != null) {
        outState.putString("saved_text", editText.getText().toString());
    }
}
```

### 3. Proper Resource Management
```java
@Override
public void onDestroyView() {
    super.onDestroyView();
    
    // Clean up to prevent memory leaks
    if (disposable != null && !disposable.isDisposed()) {
        disposable.dispose();
    }
    
    // Remove callbacks
    if (handler != null) {
        handler.removeCallbacksAndMessages(null);
    }
    
    // Nullify view references if holding them
    titleText = null;
    actionButton = null;
}
```

### 4. Use Safe Args for Navigation
```xml
<!-- Add to app/build.gradle -->
plugins {
    id 'androidx.navigation.safeargs'
}
```

```xml
<!-- navigation.xml -->
<fragment
    android:id="@+id/userFragment"
    android:name="com.example.UserFragment">
    <argument
        android:name="userId"
        app:argType="string" />
    <argument
        android:name="userName"
        app:argType="string" />
</fragment>
```

```java
// Navigate with Safe Args
UserFragmentDirections.ActionToUserFragment action = 
    UserFragmentDirections.actionToUserFragment("123", "John Doe");
Navigation.findNavController(view).navigate(action);

// Receive arguments
UserFragmentArgs args = UserFragmentArgs.fromBundle(getArguments());
String userId = args.getUserId();
String userName = args.getUserName();
```

### 5. Avoid Fragment Transactions in Async Callbacks
```java
// Wrong - can cause IllegalStateException
new AsyncTask<Void, Void, String>() {
    @Override
    protected String doInBackground(Void... voids) {
        return loadData();
    }
    
    @Override
    protected void onPostExecute(String result) {
        // This might crash if activity is destroyed
        getSupportFragmentManager()
            .beginTransaction()
            .replace(R.id.container, new ResultFragment())
            .commit();
    }
}.execute();

// Correct - check if activity is alive
@Override
protected void onPostExecute(String result) {
    if (isAdded() && !isDetached() && getActivity() != null 
            && !getActivity().isFinishing()) {
        getSupportFragmentManager()
            .beginTransaction()
            .replace(R.id.container, new ResultFragment())
            .commit();
    }
}
```

### 6. Use commitAllowingStateLoss Carefully
```java
// Use when state loss is acceptable
getSupportFragmentManager()
    .beginTransaction()
    .replace(R.id.container, fragment)
    .commitAllowingStateLoss();

// Better: Use executePendingTransactions if needed
getSupportFragmentManager()
    .beginTransaction()
    .replace(R.id.container, fragment)
    .commit();
getSupportFragmentManager().executePendingTransactions();
```

Fragments are powerful components that enable modular and flexible UI design in Android applications. Understanding their lifecycle, communication patterns, and best practices is essential for building maintainable and responsive apps.
