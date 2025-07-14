# Material Design and UI

## Table of Contents
- [Material Design Principles](#material-design-principles)
- [Material Components](#material-components)
- [Color System](#color-system)
- [Typography](#typography)
- [Theming](#theming)
- [Layouts and Spacing](#layouts-and-spacing)
- [Navigation Patterns](#navigation-patterns)
- [Animations and Transitions](#animations-and-transitions)
- [Best Practices](#best-practices)

## Material Design Principles

### Core Principles
1. **Material is the metaphor** - Inspired by physical materials
2. **Bold, graphic, intentional** - Strong visual hierarchy
3. **Motion provides meaning** - Smooth and meaningful animations

### Design Goals
- Create a unified experience across platforms
- Provide clear visual hierarchy
- Ensure accessibility and usability
- Support brand expression

## Material Components

### Add Material Components Dependency
```gradle
// app/build.gradle
dependencies {
    implementation 'com.google.android.material:material:1.10.0'
}
```

### Material Theme
```xml
<!-- res/values/themes.xml -->
<resources xmlns:tools="http://schemas.android.com/tools">
    <style name="Theme.MyApp" parent="Theme.Material3.DayNight">
        <!-- Primary brand color -->
        <item name="colorPrimary">@color/purple_500</item>
        <item name="colorPrimaryVariant">@color/purple_700</item>
        <item name="colorOnPrimary">@color/white</item>
        
        <!-- Secondary brand color -->
        <item name="colorSecondary">@color/teal_200</item>
        <item name="colorSecondaryVariant">@color/teal_700</item>
        <item name="colorOnSecondary">@color/black</item>
        
        <!-- Status bar color -->
        <item name="android:statusBarColor" tools:targetApi="l">?attr/colorPrimaryVariant</item>
        
        <!-- Customize your theme here -->
        <item name="materialButtonStyle">@style/Widget.App.Button</item>
        <item name="textInputStyle">@style/Widget.App.TextInputLayout</item>
    </style>
    
    <style name="Theme.MyApp.NoActionBar">
        <item name="windowActionBar">false</item>
        <item name="windowNoTitle">true</item>
    </style>
</resources>
```

### Material Buttons
```xml
<!-- res/layout/buttons_example.xml -->
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <!-- Filled Button (Primary) -->
    <com.google.android.material.button.MaterialButton
        android:id="@+id/btnFilled"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginBottom="8dp"
        android:text="Filled Button"
        app:icon="@drawable/ic_add"
        app:iconGravity="start" />

    <!-- Outlined Button -->
    <com.google.android.material.button.MaterialButton
        android:id="@+id/btnOutlined"
        style="@style/Widget.Material3.Button.OutlinedButton"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginBottom="8dp"
        android:text="Outlined Button"
        app:icon="@drawable/ic_edit" />

    <!-- Text Button -->
    <com.google.android.material.button.MaterialButton
        android:id="@+id/btnText"
        style="@style/Widget.Material3.Button.TextButton"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginBottom="8dp"
        android:text="Text Button" />

    <!-- Extended FAB -->
    <com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton
        android:id="@+id/extendedFab"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginBottom="8dp"
        android:text="Extended FAB"
        app:icon="@drawable/ic_add" />

    <!-- Toggle Button Group -->
    <com.google.android.material.button.MaterialButtonToggleGroup
        android:id="@+id/toggleButtonGroup"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginBottom="8dp"
        app:singleSelection="true">

        <Button
            style="?attr/materialButtonOutlinedStyle"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Option 1" />

        <Button
            style="?attr/materialButtonOutlinedStyle"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Option 2" />

        <Button
            style="?attr/materialButtonOutlinedStyle"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Option 3" />

    </com.google.android.material.button.MaterialButtonToggleGroup>

</LinearLayout>
```

### Material Cards
```xml
<!-- res/layout/cards_example.xml -->
<ScrollView xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical">

        <!-- Basic Card -->
        <com.google.android.material.card.MaterialCardView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginBottom="16dp"
            app:cardCornerRadius="8dp"
            app:cardElevation="4dp">

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="vertical"
                android:padding="16dp">

                <TextView
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:text="Card Title"
                    android:textAppearance="?attr/textAppearanceHeadline6" />

                <TextView
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:layout_marginTop="8dp"
                    android:text="Card content goes here. This is a simple card with some text content."
                    android:textAppearance="?attr/textAppearanceBody2" />

            </LinearLayout>

        </com.google.android.material.card.MaterialCardView>

        <!-- Card with Image -->
        <com.google.android.material.card.MaterialCardView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginBottom="16dp"
            app:cardCornerRadius="8dp"
            app:cardElevation="4dp">

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="vertical">

                <ImageView
                    android:layout_width="match_parent"
                    android:layout_height="200dp"
                    android:scaleType="centerCrop"
                    android:src="@drawable/sample_image" />

                <LinearLayout
                    android:layout_width="match_parent"
                    android:layout_height="wrap_content"
                    android:orientation="vertical"
                    android:padding="16dp">

                    <TextView
                        android:layout_width="wrap_content"
                        android:layout_height="wrap_content"
                        android:text="Image Card"
                        android:textAppearance="?attr/textAppearanceHeadline6" />

                    <TextView
                        android:layout_width="wrap_content"
                        android:layout_height="wrap_content"
                        android:layout_marginTop="8dp"
                        android:text="This card contains an image and text content."
                        android:textAppearance="?attr/textAppearanceBody2" />

                    <LinearLayout
                        android:layout_width="match_parent"
                        android:layout_height="wrap_content"
                        android:layout_marginTop="16dp"
                        android:orientation="horizontal">

                        <Button
                            style="@style/Widget.Material3.Button.TextButton"
                            android:layout_width="wrap_content"
                            android:layout_height="wrap_content"
                            android:text="Action 1" />

                        <Button
                            style="@style/Widget.Material3.Button.TextButton"
                            android:layout_width="wrap_content"
                            android:layout_height="wrap_content"
                            android:text="Action 2" />

                    </LinearLayout>

                </LinearLayout>

            </LinearLayout>

        </com.google.android.material.card.MaterialCardView>

        <!-- Outlined Card -->
        <com.google.android.material.card.MaterialCardView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginBottom="16dp"
            app:cardCornerRadius="8dp"
            app:cardElevation="0dp"
            app:strokeColor="?attr/colorOutline"
            app:strokeWidth="1dp">

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="vertical"
                android:padding="16dp">

                <TextView
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:text="Outlined Card"
                    android:textAppearance="?attr/textAppearanceHeadline6" />

                <TextView
                    android:layout_width="wrap_content"
                    android:layout_height="wrap_content"
                    android:layout_marginTop="8dp"
                    android:text="This is an outlined card with stroke border instead of elevation."
                    android:textAppearance="?attr/textAppearanceBody2" />

            </LinearLayout>

        </com.google.android.material.card.MaterialCardView>

    </LinearLayout>

</ScrollView>
```

### Text Input Fields
```xml
<!-- res/layout/text_inputs_example.xml -->
<ScrollView xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical">

        <!-- Filled Text Field -->
        <com.google.android.material.textfield.TextInputLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginBottom="16dp"
            android:hint="Filled Text Field"
            app:startIconDrawable="@drawable/ic_person">

            <com.google.android.material.textfield.TextInputEditText
                android:layout_width="match_parent"
                android:layout_height="wrap_content" />

        </com.google.android.material.textfield.TextInputLayout>

        <!-- Outlined Text Field -->
        <com.google.android.material.textfield.TextInputLayout
            style="@style/Widget.Material3.TextInputLayout.OutlinedBox"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginBottom="16dp"
            android:hint="Outlined Text Field"
            app:endIconMode="clear_text"
            app:startIconDrawable="@drawable/ic_email">

            <com.google.android.material.textfield.TextInputEditText
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:inputType="textEmailAddress" />

        </com.google.android.material.textfield.TextInputLayout>

        <!-- Password Field -->
        <com.google.android.material.textfield.TextInputLayout
            style="@style/Widget.Material3.TextInputLayout.OutlinedBox"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginBottom="16dp"
            android:hint="Password"
            app:endIconMode="password_toggle"
            app:startIconDrawable="@drawable/ic_lock">

            <com.google.android.material.textfield.TextInputEditText
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:inputType="textPassword" />

        </com.google.android.material.textfield.TextInputLayout>

        <!-- Dropdown Menu -->
        <com.google.android.material.textfield.TextInputLayout
            style="@style/Widget.Material3.TextInputLayout.OutlinedBox.ExposedDropdownMenu"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginBottom="16dp"
            android:hint="Dropdown">

            <AutoCompleteTextView
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:inputType="none" />

        </com.google.android.material.textfield.TextInputLayout>

        <!-- Multiline Text Field -->
        <com.google.android.material.textfield.TextInputLayout
            style="@style/Widget.Material3.TextInputLayout.OutlinedBox"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginBottom="16dp"
            android:hint="Message"
            app:counterEnabled="true"
            app:counterMaxLength="500">

            <com.google.android.material.textfield.TextInputEditText
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:inputType="textMultiLine"
                android:lines="4"
                android:maxLength="500" />

        </com.google.android.material.textfield.TextInputLayout>

    </LinearLayout>

</ScrollView>
```

### Material App Bar
```xml
<!-- res/layout/app_bar_example.xml -->
<androidx.coordinatorlayout.widget.CoordinatorLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <com.google.android.material.appbar.AppBarLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content">

        <!-- Collapsing Toolbar -->
        <com.google.android.material.appbar.CollapsingToolbarLayout
            android:layout_width="match_parent"
            android:layout_height="200dp"
            app:contentScrim="?attr/colorPrimary"
            app:layout_scrollFlags="scroll|exitUntilCollapsed"
            app:title="Collapsing Toolbar">

            <ImageView
                android:layout_width="match_parent"
                android:layout_height="match_parent"
                android:scaleType="centerCrop"
                android:src="@drawable/header_image"
                app:layout_collapseMode="parallax" />

            <androidx.appcompat.widget.Toolbar
                android:id="@+id/toolbar"
                android:layout_width="match_parent"
                android:layout_height="?attr/actionBarSize"
                app:layout_collapseMode="pin"
                app:popupTheme="@style/ThemeOverlay.Material3.Light" />

        </com.google.android.material.appbar.CollapsingToolbarLayout>

    </com.google.android.material.appbar.AppBarLayout>

    <!-- Main content -->
    <androidx.core.widget.NestedScrollView
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        app:layout_behavior="@string/appbar_scrolling_view_behavior">

        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="vertical"
            android:padding="16dp">

            <!-- Content goes here -->
            <TextView
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="Main content"
                android:textAppearance="?attr/textAppearanceHeadline6" />

        </LinearLayout>

    </androidx.core.widget.NestedScrollView>

    <!-- FAB -->
    <com.google.android.material.floatingactionbutton.FloatingActionButton
        android:id="@+id/fab"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_margin="16dp"
        android:src="@drawable/ic_add"
        app:layout_anchor="@id/toolbar"
        app:layout_anchorGravity="bottom|end" />

</androidx.coordinatorlayout.widget.CoordinatorLayout>
```

## Color System

### Material 3 Color Tokens
```xml
<!-- res/values/colors.xml -->
<resources>
    <!-- Material 3 Color System -->
    
    <!-- Primary colors -->
    <color name="md_theme_light_primary">#6750A4</color>
    <color name="md_theme_light_onPrimary">#FFFFFF</color>
    <color name="md_theme_light_primaryContainer">#EADDFF</color>
    <color name="md_theme_light_onPrimaryContainer">#21005D</color>
    
    <!-- Secondary colors -->
    <color name="md_theme_light_secondary">#625B71</color>
    <color name="md_theme_light_onSecondary">#FFFFFF</color>
    <color name="md_theme_light_secondaryContainer">#E8DEF8</color>
    <color name="md_theme_light_onSecondaryContainer">#1D192B</color>
    
    <!-- Tertiary colors -->
    <color name="md_theme_light_tertiary">#7D5260</color>
    <color name="md_theme_light_onTertiary">#FFFFFF</color>
    <color name="md_theme_light_tertiaryContainer">#FFD8E4</color>
    <color name="md_theme_light_onTertiaryContainer">#31111D</color>
    
    <!-- Error colors -->
    <color name="md_theme_light_error">#BA1A1A</color>
    <color name="md_theme_light_errorContainer">#FFDAD6</color>
    <color name="md_theme_light_onError">#FFFFFF</color>
    <color name="md_theme_light_onErrorContainer">#410002</color>
    
    <!-- Surface colors -->
    <color name="md_theme_light_background">#FFFBFE</color>
    <color name="md_theme_light_onBackground">#1C1B1F</color>
    <color name="md_theme_light_surface">#FFFBFE</color>
    <color name="md_theme_light_onSurface">#1C1B1F</color>
    <color name="md_theme_light_surfaceVariant">#E7E0EC</color>
    <color name="md_theme_light_onSurfaceVariant">#49454F</color>
    
    <!-- Dark theme colors -->
    <color name="md_theme_dark_primary">#D0BCFF</color>
    <color name="md_theme_dark_onPrimary">#381E72</color>
    <color name="md_theme_dark_primaryContainer">#4F378B</color>
    <color name="md_theme_dark_onPrimaryContainer">#EADDFF</color>
    
    <!-- Add other dark theme colors... -->
    
</resources>
```

### Dynamic Color Theme
```java
public class ColorSystemManager {
    
    public static void applyDynamicColor(Activity activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            // Material You dynamic color (Android 12+)
            DynamicColors.applyToActivityIfAvailable(activity);
        }
    }
    
    public static void applyCustomColorScheme(Activity activity, boolean isDarkMode) {
        int themeId = isDarkMode ? R.style.Theme_MyApp_Dark : R.style.Theme_MyApp_Light;
        activity.setTheme(themeId);
    }
    
    @ColorInt
    public static int getThemeColor(Context context, @AttrRes int attributeId) {
        TypedValue typedValue = new TypedValue();
        context.getTheme().resolveAttribute(attributeId, typedValue, true);
        return typedValue.data;
    }
    
    public static boolean isDarkModeEnabled(Context context) {
        int nightModeFlags = context.getResources().getConfiguration().uiMode 
                           & Configuration.UI_MODE_NIGHT_MASK;
        return nightModeFlags == Configuration.UI_MODE_NIGHT_YES;
    }
}
```

## Typography

### Material 3 Typography Scale
```xml
<!-- res/values/type.xml -->
<resources>
    
    <!-- Material 3 Typography Scale -->
    <style name="TextAppearance.App.DisplayLarge" parent="TextAppearance.Material3.DisplayLarge">
        <item name="fontFamily">@font/roboto</item>
        <item name="android:fontWeight">400</item>
        <item name="android:textSize">57sp</item>
        <item name="android:letterSpacing">-0.25sp</item>
        <item name="android:lineHeight">64sp</item>
    </style>
    
    <style name="TextAppearance.App.DisplayMedium" parent="TextAppearance.Material3.DisplayMedium">
        <item name="fontFamily">@font/roboto</item>
        <item name="android:fontWeight">400</item>
        <item name="android:textSize">45sp</item>
        <item name="android:letterSpacing">0sp</item>
        <item name="android:lineHeight">52sp</item>
    </style>
    
    <style name="TextAppearance.App.DisplaySmall" parent="TextAppearance.Material3.DisplaySmall">
        <item name="fontFamily">@font/roboto</item>
        <item name="android:fontWeight">400</item>
        <item name="android:textSize">36sp</item>
        <item name="android:letterSpacing">0sp</item>
        <item name="android:lineHeight">44sp</item>
    </style>
    
    <style name="TextAppearance.App.HeadlineLarge" parent="TextAppearance.Material3.HeadlineLarge">
        <item name="fontFamily">@font/roboto</item>
        <item name="android:fontWeight">400</item>
        <item name="android:textSize">32sp</item>
        <item name="android:letterSpacing">0sp</item>
        <item name="android:lineHeight">40sp</item>
    </style>
    
    <style name="TextAppearance.App.HeadlineMedium" parent="TextAppearance.Material3.HeadlineMedium">
        <item name="fontFamily">@font/roboto</item>
        <item name="android:fontWeight">400</item>
        <item name="android:textSize">28sp</item>
        <item name="android:letterSpacing">0sp</item>
        <item name="android:lineHeight">36sp</item>
    </style>
    
    <style name="TextAppearance.App.HeadlineSmall" parent="TextAppearance.Material3.HeadlineSmall">
        <item name="fontFamily">@font/roboto</item>
        <item name="android:fontWeight">400</item>
        <item name="android:textSize">24sp</item>
        <item name="android:letterSpacing">0sp</item>
        <item name="android:lineHeight">32sp</item>
    </style>
    
    <style name="TextAppearance.App.TitleLarge" parent="TextAppearance.Material3.TitleLarge">
        <item name="fontFamily">@font/roboto</item>
        <item name="android:fontWeight">400</item>
        <item name="android:textSize">22sp</item>
        <item name="android:letterSpacing">0sp</item>
        <item name="android:lineHeight">28sp</item>
    </style>
    
    <style name="TextAppearance.App.TitleMedium" parent="TextAppearance.Material3.TitleMedium">
        <item name="fontFamily">@font/roboto_medium</item>
        <item name="android:fontWeight">500</item>
        <item name="android:textSize">16sp</item>
        <item name="android:letterSpacing">0.15sp</item>
        <item name="android:lineHeight">24sp</item>
    </style>
    
    <style name="TextAppearance.App.TitleSmall" parent="TextAppearance.Material3.TitleSmall">
        <item name="fontFamily">@font/roboto_medium</item>
        <item name="android:fontWeight">500</item>
        <item name="android:textSize">14sp</item>
        <item name="android:letterSpacing">0.1sp</item>
        <item name="android:lineHeight">20sp</item>
    </style>
    
    <style name="TextAppearance.App.LabelLarge" parent="TextAppearance.Material3.LabelLarge">
        <item name="fontFamily">@font/roboto_medium</item>
        <item name="android:fontWeight">500</item>
        <item name="android:textSize">14sp</item>
        <item name="android:letterSpacing">0.1sp</item>
        <item name="android:lineHeight">20sp</item>
    </style>
    
    <style name="TextAppearance.App.LabelMedium" parent="TextAppearance.Material3.LabelMedium">
        <item name="fontFamily">@font/roboto_medium</item>
        <item name="android:fontWeight">500</item>
        <item name="android:textSize">12sp</item>
        <item name="android:letterSpacing">0.5sp</item>
        <item name="android:lineHeight">16sp</item>
    </style>
    
    <style name="TextAppearance.App.LabelSmall" parent="TextAppearance.Material3.LabelSmall">
        <item name="fontFamily">@font/roboto_medium</item>
        <item name="android:fontWeight">500</item>
        <item name="android:textSize">11sp</item>
        <item name="android:letterSpacing">0.5sp</item>
        <item name="android:lineHeight">16sp</item>
    </style>
    
    <style name="TextAppearance.App.BodyLarge" parent="TextAppearance.Material3.BodyLarge">
        <item name="fontFamily">@font/roboto</item>
        <item name="android:fontWeight">400</item>
        <item name="android:textSize">16sp</item>
        <item name="android:letterSpacing">0.5sp</item>
        <item name="android:lineHeight">24sp</item>
    </style>
    
    <style name="TextAppearance.App.BodyMedium" parent="TextAppearance.Material3.BodyMedium">
        <item name="fontFamily">@font/roboto</item>
        <item name="android:fontWeight">400</item>
        <item name="android:textSize">14sp</item>
        <item name="android:letterSpacing">0.25sp</item>
        <item name="android:lineHeight">20sp</item>
    </style>
    
    <style name="TextAppearance.App.BodySmall" parent="TextAppearance.Material3.BodySmall">
        <item name="fontFamily">@font/roboto</item>
        <item name="android:fontWeight">400</item>
        <item name="android:textSize">12sp</item>
        <item name="android:letterSpacing">0.4sp</item>
        <item name="android:lineHeight">16sp</item>
    </style>
    
</resources>
```

### Custom Fonts
```xml
<!-- res/font/font_family.xml -->
<?xml version="1.0" encoding="utf-8"?>
<font-family xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto">
    
    <font
        android:fontStyle="normal"
        android:fontWeight="400"
        android:font="@font/roboto_regular"
        app:fontStyle="normal"
        app:fontWeight="400"
        app:font="@font/roboto_regular" />
    
    <font
        android:fontStyle="normal"
        android:fontWeight="500"
        android:font="@font/roboto_medium"
        app:fontStyle="normal"
        app:fontWeight="500"
        app:font="@font/roboto_medium" />
    
    <font
        android:fontStyle="normal"
        android:fontWeight="700"
        android:font="@font/roboto_bold"
        app:fontStyle="normal"
        app:fontWeight="700"
        app:font="@font/roboto_bold" />
        
</font-family>
```

## Theming

### Material 3 Theme
```xml
<!-- res/values/themes.xml -->
<resources xmlns:tools="http://schemas.android.com/tools">
    
    <style name="Theme.MyApp" parent="Theme.Material3.DayNight">
        <!-- Primary colors -->
        <item name="colorPrimary">@color/md_theme_light_primary</item>
        <item name="colorOnPrimary">@color/md_theme_light_onPrimary</item>
        <item name="colorPrimaryContainer">@color/md_theme_light_primaryContainer</item>
        <item name="colorOnPrimaryContainer">@color/md_theme_light_onPrimaryContainer</item>
        
        <!-- Secondary colors -->
        <item name="colorSecondary">@color/md_theme_light_secondary</item>
        <item name="colorOnSecondary">@color/md_theme_light_onSecondary</item>
        <item name="colorSecondaryContainer">@color/md_theme_light_secondaryContainer</item>
        <item name="colorOnSecondaryContainer">@color/md_theme_light_onSecondaryContainer</item>
        
        <!-- Tertiary colors -->
        <item name="colorTertiary">@color/md_theme_light_tertiary</item>
        <item name="colorOnTertiary">@color/md_theme_light_onTertiary</item>
        <item name="colorTertiaryContainer">@color/md_theme_light_tertiaryContainer</item>
        <item name="colorOnTertiaryContainer">@color/md_theme_light_onTertiaryContainer</item>
        
        <!-- Error colors -->
        <item name="colorError">@color/md_theme_light_error</item>
        <item name="colorErrorContainer">@color/md_theme_light_errorContainer</item>
        <item name="colorOnError">@color/md_theme_light_onError</item>
        <item name="colorOnErrorContainer">@color/md_theme_light_onErrorContainer</item>
        
        <!-- Surface colors -->
        <item name="android:colorBackground">@color/md_theme_light_background</item>
        <item name="colorOnBackground">@color/md_theme_light_onBackground</item>
        <item name="colorSurface">@color/md_theme_light_surface</item>
        <item name="colorOnSurface">@color/md_theme_light_onSurface</item>
        <item name="colorSurfaceVariant">@color/md_theme_light_surfaceVariant</item>
        <item name="colorOnSurfaceVariant">@color/md_theme_light_onSurfaceVariant</item>
        
        <!-- Status bar -->
        <item name="android:statusBarColor" tools:targetApi="l">?attr/colorPrimary</item>
        
        <!-- Component themes -->
        <item name="materialButtonStyle">@style/Widget.App.Button</item>
        <item name="textInputStyle">@style/Widget.App.TextInputLayout</item>
        <item name="materialCardViewStyle">@style/Widget.App.CardView</item>
        
        <!-- Typography -->
        <item name="textAppearanceDisplayLarge">@style/TextAppearance.App.DisplayLarge</item>
        <item name="textAppearanceDisplayMedium">@style/TextAppearance.App.DisplayMedium</item>
        <item name="textAppearanceDisplaySmall">@style/TextAppearance.App.DisplaySmall</item>
        <item name="textAppearanceHeadlineLarge">@style/TextAppearance.App.HeadlineLarge</item>
        <item name="textAppearanceHeadlineMedium">@style/TextAppearance.App.HeadlineMedium</item>
        <item name="textAppearanceHeadlineSmall">@style/TextAppearance.App.HeadlineSmall</item>
        <item name="textAppearanceTitleLarge">@style/TextAppearance.App.TitleLarge</item>
        <item name="textAppearanceTitleMedium">@style/TextAppearance.App.TitleMedium</item>
        <item name="textAppearanceTitleSmall">@style/TextAppearance.App.TitleSmall</item>
        <item name="textAppearanceLabelLarge">@style/TextAppearance.App.LabelLarge</item>
        <item name="textAppearanceLabelMedium">@style/TextAppearance.App.LabelMedium</item>
        <item name="textAppearanceLabelSmall">@style/TextAppearance.App.LabelSmall</item>
        <item name="textAppearanceBodyLarge">@style/TextAppearance.App.BodyLarge</item>
        <item name="textAppearanceBodyMedium">@style/TextAppearance.App.BodyMedium</item>
        <item name="textAppearanceBodySmall">@style/TextAppearance.App.BodySmall</item>
        
    </style>
    
    <!-- Custom component styles -->
    <style name="Widget.App.Button" parent="Widget.Material3.Button">
        <item name="android:textAppearance">?attr/textAppearanceLabelLarge</item>
        <item name="cornerRadius">8dp</item>
    </style>
    
    <style name="Widget.App.Button.Outlined" parent="Widget.Material3.Button.OutlinedButton">
        <item name="android:textAppearance">?attr/textAppearanceLabelLarge</item>
        <item name="cornerRadius">8dp</item>
    </style>
    
    <style name="Widget.App.TextInputLayout" parent="Widget.Material3.TextInputLayout.OutlinedBox">
        <item name="boxCornerRadiusTopStart">8dp</item>
        <item name="boxCornerRadiusTopEnd">8dp</item>
        <item name="boxCornerRadiusBottomStart">8dp</item>
        <item name="boxCornerRadiusBottomEnd">8dp</item>
    </style>
    
    <style name="Widget.App.CardView" parent="Widget.Material3.CardView.Elevated">
        <item name="cardCornerRadius">12dp</item>
        <item name="cardElevation">4dp</item>
    </style>
    
</resources>
```

### Dark Theme
```xml
<!-- res/values-night/themes.xml -->
<resources xmlns:tools="http://schemas.android.com/tools">
    
    <style name="Theme.MyApp" parent="Theme.Material3.DayNight">
        <!-- Dark theme colors -->
        <item name="colorPrimary">@color/md_theme_dark_primary</item>
        <item name="colorOnPrimary">@color/md_theme_dark_onPrimary</item>
        <item name="colorPrimaryContainer">@color/md_theme_dark_primaryContainer</item>
        <item name="colorOnPrimaryContainer">@color/md_theme_dark_onPrimaryContainer</item>
        
        <!-- Add other dark theme color mappings... -->
        
        <item name="android:statusBarColor" tools:targetApi="l">?attr/colorPrimary</item>
    </style>
    
</resources>
```

### Theme Manager
```java
public class ThemeManager {
    
    private static final String PREFS_NAME = "theme_prefs";
    private static final String KEY_THEME_MODE = "theme_mode";
    
    public enum ThemeMode {
        LIGHT, DARK, SYSTEM
    }
    
    public static void applyTheme(Activity activity) {
        ThemeMode mode = getThemeMode(activity);
        
        switch (mode) {
            case LIGHT:
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO);
                break;
            case DARK:
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES);
                break;
            case SYSTEM:
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM);
                break;
        }
        
        // Apply dynamic colors if available
        ColorSystemManager.applyDynamicColor(activity);
    }
    
    public static void setThemeMode(Context context, ThemeMode mode) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        prefs.edit().putString(KEY_THEME_MODE, mode.name()).apply();
        
        // Apply immediately
        switch (mode) {
            case LIGHT:
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO);
                break;
            case DARK:
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES);
                break;
            case SYSTEM:
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM);
                break;
        }
    }
    
    public static ThemeMode getThemeMode(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        String themeName = prefs.getString(KEY_THEME_MODE, ThemeMode.SYSTEM.name());
        return ThemeMode.valueOf(themeName);
    }
    
    public static boolean isDarkModeActive(Context context) {
        int nightModeFlags = context.getResources().getConfiguration().uiMode 
                           & Configuration.UI_MODE_NIGHT_MASK;
        return nightModeFlags == Configuration.UI_MODE_NIGHT_YES;
    }
}
```

## Navigation Patterns

### Bottom Navigation
```xml
<!-- res/layout/activity_main_bottom_nav.xml -->
<androidx.coordinatorlayout.widget.CoordinatorLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <FrameLayout
        android:id="@+id/nav_host_fragment"
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        android:layout_marginBottom="?attr/actionBarSize" />

    <com.google.android.material.bottomnavigation.BottomNavigationView
        android:id="@+id/bottom_navigation"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_gravity="bottom"
        app:menu="@menu/bottom_navigation" />

</androidx.coordinatorlayout.widget.CoordinatorLayout>
```

```xml
<!-- res/menu/bottom_navigation.xml -->
<menu xmlns:android="http://schemas.android.com/apk/res/android">
    <item
        android:id="@+id/navigation_home"
        android:icon="@drawable/ic_home"
        android:title="Home" />
    
    <item
        android:id="@+id/navigation_search"
        android:icon="@drawable/ic_search"
        android:title="Search" />
    
    <item
        android:id="@+id/navigation_favorites"
        android:icon="@drawable/ic_favorite"
        android:title="Favorites" />
    
    <item
        android:id="@+id/navigation_profile"
        android:icon="@drawable/ic_person"
        android:title="Profile" />
</menu>
```

### Navigation Drawer
```xml
<!-- res/layout/activity_main_drawer.xml -->
<androidx.drawerlayout.widget.DrawerLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:id="@+id/drawer_layout"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:fitsSystemWindows="true"
    tools:openDrawer="start">

    <!-- Main content -->
    <androidx.coordinatorlayout.widget.CoordinatorLayout
        android:layout_width="match_parent"
        android:layout_height="match_parent">

        <com.google.android.material.appbar.AppBarLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content">

            <androidx.appcompat.widget.Toolbar
                android:id="@+id/toolbar"
                android:layout_width="match_parent"
                android:layout_height="?attr/actionBarSize"
                android:background="?attr/colorPrimary"
                app:popupTheme="@style/ThemeOverlay.Material3.Light" />

        </com.google.android.material.appbar.AppBarLayout>

        <FrameLayout
            android:id="@+id/nav_host_fragment"
            android:layout_width="match_parent"
            android:layout_height="match_parent"
            app:layout_behavior="@string/appbar_scrolling_view_behavior" />

    </androidx.coordinatorlayout.widget.CoordinatorLayout>

    <!-- Navigation drawer -->
    <com.google.android.material.navigation.NavigationView
        android:id="@+id/nav_view"
        android:layout_width="wrap_content"
        android:layout_height="match_parent"
        android:layout_gravity="start"
        android:fitsSystemWindows="true"
        app:headerLayout="@layout/nav_header_main"
        app:menu="@menu/activity_main_drawer" />

</androidx.drawerlayout.widget.DrawerLayout>
```

### Tab Layout
```xml
<!-- res/layout/activity_tabs.xml -->
<androidx.coordinatorlayout.widget.CoordinatorLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <com.google.android.material.appbar.AppBarLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content">

        <androidx.appcompat.widget.Toolbar
            android:id="@+id/toolbar"
            android:layout_width="match_parent"
            android:layout_height="?attr/actionBarSize"
            app:popupTheme="@style/ThemeOverlay.Material3.Light" />

        <com.google.android.material.tabs.TabLayout
            android:id="@+id/tabs"
            android:layout_width="match_parent"
            android:layout_height="wrap_content" />

    </com.google.android.material.appbar.AppBarLayout>

    <androidx.viewpager2.widget.ViewPager2
        android:id="@+id/view_pager"
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        app:layout_behavior="@string/appbar_scrolling_view_behavior" />

</androidx.coordinatorlayout.widget.CoordinatorLayout>
```

## Animations and Transitions

### Shared Element Transitions
```java
public class TransitionHelper {
    
    public static void startActivityWithTransition(Activity activity, Intent intent, View sharedElement, String transitionName) {
        ActivityOptionsCompat options = ActivityOptionsCompat.makeSceneTransitionAnimation(
            activity, sharedElement, transitionName);
        ActivityCompat.startActivity(activity, intent, options.toBundle());
    }
    
    public static void setupSharedElementTransitions(Activity activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            // Set up enter and exit transitions
            Slide slide = new Slide(Gravity.BOTTOM);
            slide.setDuration(300);
            activity.getWindow().setEnterTransition(slide);
            activity.getWindow().setExitTransition(slide);
            
            // Set up shared element transitions
            ChangeBounds changeBounds = new ChangeBounds();
            changeBounds.setDuration(300);
            activity.getWindow().setSharedElementEnterTransition(changeBounds);
            activity.getWindow().setSharedElementExitTransition(changeBounds);
        }
    }
}
```

### Material Motion
```java
public class MaterialMotion {
    
    public static void fadeInView(View view) {
        view.setAlpha(0f);
        view.setVisibility(View.VISIBLE);
        view.animate()
            .alpha(1f)
            .setDuration(300)
            .setInterpolator(new DecelerateInterpolator())
            .start();
    }
    
    public static void slideInFromBottom(View view) {
        view.setTranslationY(view.getHeight());
        view.setVisibility(View.VISIBLE);
        view.animate()
            .translationY(0)
            .setDuration(300)
            .setInterpolator(new DecelerateInterpolator())
            .start();
    }
    
    public static void scaleIn(View view) {
        view.setScaleX(0f);
        view.setScaleY(0f);
        view.setVisibility(View.VISIBLE);
        view.animate()
            .scaleX(1f)
            .scaleY(1f)
            .setDuration(300)
            .setInterpolator(new OvershootInterpolator())
            .start();
    }
    
    public static void morphButton(View fromView, View toView) {
        // Create circular reveal animation
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            int cx = toView.getWidth() / 2;
            int cy = toView.getHeight() / 2;
            float finalRadius = (float) Math.hypot(cx, cy);
            
            Animator anim = ViewAnimationUtils.createCircularReveal(toView, cx, cy, 0, finalRadius);
            anim.setDuration(300);
            
            toView.setVisibility(View.VISIBLE);
            anim.start();
            
            // Hide original view
            fromView.animate()
                .alpha(0f)
                .setDuration(150)
                .withEndAction(() -> fromView.setVisibility(View.GONE))
                .start();
        }
    }
}
```

## Best Practices

### 1. Accessibility
```java
public class AccessibilityHelper {
    
    public static void setupAccessibility(View view, String contentDescription, String hint) {
        view.setContentDescription(contentDescription);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            view.setStateDescription(hint);
        }
        view.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_YES);
    }
    
    public static void announceForAccessibility(View view, String message) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN) {
            view.announceForAccessibility(message);
        }
    }
    
    public static void setAccessibilityHeading(View view, boolean isHeading) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            view.setAccessibilityHeading(isHeading);
        }
    }
    
    public static boolean isAccessibilityEnabled(Context context) {
        AccessibilityManager am = (AccessibilityManager) context.getSystemService(Context.ACCESSIBILITY_SERVICE);
        return am != null && am.isEnabled();
    }
}
```

### 2. Responsive Design
```java
public class ResponsiveHelper {
    
    public static boolean isTablet(Context context) {
        return (context.getResources().getConfiguration().screenLayout 
                & Configuration.SCREENLAYOUT_SIZE_MASK) >= Configuration.SCREENLAYOUT_SIZE_LARGE;
    }
    
    public static int getScreenWidth(Context context) {
        DisplayMetrics displayMetrics = context.getResources().getDisplayMetrics();
        return displayMetrics.widthPixels;
    }
    
    public static int dpToPx(Context context, int dp) {
        float density = context.getResources().getDisplayMetrics().density;
        return Math.round(dp * density);
    }
    
    public static int getColumnCount(Context context, int itemWidth) {
        int screenWidth = getScreenWidth(context);
        int itemWidthPx = dpToPx(context, itemWidth);
        return Math.max(1, screenWidth / itemWidthPx);
    }
}
```

### 3. Performance Optimization
```java
public class UIPerformanceHelper {
    
    public static void optimizeRecyclerView(RecyclerView recyclerView) {
        recyclerView.setHasFixedSize(true);
        recyclerView.setItemAnimator(null); // Disable if not needed
        recyclerView.setDrawingCacheEnabled(true);
        recyclerView.setDrawingCacheQuality(View.DRAWING_CACHE_QUALITY_HIGH);
    }
    
    public static void preloadImages(Context context, List<String> imageUrls) {
        // Use Glide to preload images
        for (String url : imageUrls) {
            Glide.with(context)
                .load(url)
                .preload();
        }
    }
    
    public static void reduceOverdraw(View view) {
        // Remove unnecessary backgrounds
        if (view.getParent() instanceof ViewGroup) {
            ViewGroup parent = (ViewGroup) view.getParent();
            if (parent.getBackground() != null) {
                view.setBackground(null);
            }
        }
    }
}
```

Material Design provides a comprehensive design system that ensures consistency, accessibility, and beautiful user interfaces. Following Material Design guidelines helps create apps that feel familiar and intuitive to users while maintaining your brand identity.
