from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Replace these with your FinViz username and password
USERNAME = 'muellerlennon@gmail.com'
PASSWORD = 'Accident55%'

# Set up the WebDriver
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()

# Specify download directory, disable download prompt, headless option
options.add_experimental_option('prefs', {
    "download.default_directory": r"C:\Users\lmueller\Desktop\Project M\Financial Data from Finviz",
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

driver = webdriver.Chrome(service=service, options=options)

# Navigate to the FinViz homepage
print("Navigating to FinViz homepage...")
driver.get('https://finviz.com')
print("Homepage loaded.")

try:
    # Click the login button to open the login form
    print("Clicking the login button...")
    login_button = WebDriverWait(driver, 10).until(
        # Adjust the selector if necessary
        EC.element_to_be_clickable((By.LINK_TEXT, "Login"))
    )
    login_button.click()

    # Wait and fill in the username and password fields
    print("Filling in login credentials...")
    WebDriverWait(driver, 10).until(EC.presence_of_element_located(
        (By.NAME, "email"))).send_keys(USERNAME)
    driver.find_element(By.NAME, "password").send_keys(PASSWORD)

    # Click the submit button to log in
    print("Submitting login form...")
    driver.find_element(By.CSS_SELECTOR, "input[type='submit']").click()

    # Navigate to the screener page
    print("Navigating to FinViz screener page...")
    driver.get('https://elite.finviz.com/screener.ashx?v=161&f=exch_nasd,idx_ndx')
    print("Screener page loaded.")

    # Wait for the export button to be present
    print("Waiting for the export button to be present...")
    wait = WebDriverWait(driver, 20)
    export_button = wait.until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "a.tab-link[href*='export']"))
    )
    print("Export button found.")

    # Click the export button using JavaScript
    print("Clicking the export button...")
    driver.execute_script("arguments[0].click();", export_button)

    # Wait for the download to start
    print("Waiting for the download to start...")
    WebDriverWait(driver, 10).until(
        lambda d: "download" in d.current_url.lower(
        ) or "elite.ashx" in d.current_url.lower()
    )
    print("Download should have started.")

except Exception as e:
    print("Error during process:", e)

finally:
    # Close the browser
    print("Closing the browser...")
    driver.quit()
    print("Browser closed.")
