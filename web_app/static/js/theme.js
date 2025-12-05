/**
 * Theme Management System
 * Handles dark/light theme switching with localStorage persistence
 */

const ThemeManager = {
    STORAGE_KEY: 'pidu-theme',
    DARK: 'dark',
    LIGHT: 'light',

    /**
     * Initialize theme on page load
     */
    init() {
        // Get saved theme or detect system preference
        const savedTheme = localStorage.getItem(this.STORAGE_KEY);
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        const theme = savedTheme || (systemPrefersDark ? this.DARK : this.LIGHT);
        this.setTheme(theme, false);
        
        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!localStorage.getItem(this.STORAGE_KEY)) {
                this.setTheme(e.matches ? this.DARK : this.LIGHT, false);
            }
        });
        
        // Update toggle button state
        this.updateToggleButton();
    },

    /**
     * Set theme
     * @param {string} theme - 'dark' or 'light'
     * @param {boolean} save - Whether to save to localStorage
     */
    setTheme(theme, save = true) {
        document.documentElement.setAttribute('data-theme', theme);
        
        if (save) {
            localStorage.setItem(this.STORAGE_KEY, theme);
        }
        
        this.updateToggleButton();
        this.updateMetaThemeColor(theme);
        
        // Dispatch custom event for components that need to react
        window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
    },

    /**
     * Toggle between dark and light theme
     */
    toggle() {
        const currentTheme = this.getCurrentTheme();
        const newTheme = currentTheme === this.DARK ? this.LIGHT : this.DARK;
        this.setTheme(newTheme);
    },

    /**
     * Get current theme
     * @returns {string} Current theme ('dark' or 'light')
     */
    getCurrentTheme() {
        return document.documentElement.getAttribute('data-theme') || this.LIGHT;
    },

    /**
     * Check if dark mode is active
     * @returns {boolean}
     */
    isDark() {
        return this.getCurrentTheme() === this.DARK;
    },

    /**
     * Update toggle button icon
     */
    updateToggleButton() {
        const toggleBtn = document.getElementById('theme-toggle');
        if (toggleBtn) {
            const isDark = this.isDark();
            const icon = toggleBtn.querySelector('i');
            if (icon) {
                icon.className = isDark ? 'bi bi-sun-fill' : 'bi bi-moon-fill';
            }
            toggleBtn.title = isDark ? 'Açık Tema' : 'Koyu Tema';
        }
    },

    /**
     * Update meta theme-color for mobile browsers
     * @param {string} theme
     */
    updateMetaThemeColor(theme) {
        let metaThemeColor = document.querySelector('meta[name="theme-color"]');
        if (!metaThemeColor) {
            metaThemeColor = document.createElement('meta');
            metaThemeColor.name = 'theme-color';
            document.head.appendChild(metaThemeColor);
        }
        metaThemeColor.content = theme === this.DARK ? '#16213e' : '#ffffff';
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
});

// Also initialize immediately if DOM is already loaded
if (document.readyState !== 'loading') {
    ThemeManager.init();
}

// Global function for onclick handlers
function toggleTheme() {
    ThemeManager.toggle();
}
