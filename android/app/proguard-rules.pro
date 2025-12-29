# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in the SDK tools proguard-defaults.txt file.

# Keep PIL core classes
-keep class com.pilvae.engine.core.** { *; }

# Keep numerical utilities
-keepclassmembers class com.pilvae.engine.core.PILUtils {
    public static ** *(...);
}
