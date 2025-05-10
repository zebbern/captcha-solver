import time
import os
import numpy as np
import cv2
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle
import shutil
import matplotlib.pyplot as plt

class FastCaptchaTrainer:
    def __init__(self):
        # Initialize webdriver with options
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        self.driver = webdriver.Chrome(options=options)
        
        # Create database directory
        os.makedirs("captcha_db", exist_ok=True)
        os.makedirs("captcha_db/samples", exist_ok=True)
        os.makedirs("captcha_db/chars", exist_ok=True)
        
        # Load existing character database if it exists
        self.char_db = {}
        self.db_file = "captcha_db/char_db.pkl"
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "rb") as f:
                    self.char_db = pickle.load(f)
                print(f"Loaded existing database with {sum(len(v) for v in self.char_db.values())} templates")
                print(f"Known characters: {', '.join(sorted(self.char_db.keys()))}")
            except:
                print("Could not load database, starting fresh")
        
        # Counter for saving samples
        self.sample_count = len(os.listdir("captcha_db/samples"))
        
        # First-time initialization flag
        self.initialized = False
        
        # 30 threshold values starting at 40 with increments of 3
        self.thresholds = [40 + (i * 3) for i in range(30)]
        
        # Training statistics
        self.trained_count = 0
        self.skipped_count = 0

    def initialize(self):
        """Do first-time initialization"""
        if self.initialized:
            return
            
        print("Performing first-time initialization...")
        
        # Navigate to the website
        self.driver.get("https://zefoy.com")
        
        # Wait for initial page load
        time.sleep(5)
        
        # Handle any popups
        self.handle_popups()
        
        # Wait for captcha to appear
        try:
            print("Waiting for CAPTCHA page to load...")
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//img[contains(@src, '_CAPTCHA')]"))
            )
        except:
            print("Trying alternate selector...")
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/div[5]/div[2]/form/div/div/img"))
            )
        
        print("Initialization complete!")
        self.initialized = True

    def handle_popups(self):
        """Handle any consent popups that might appear"""
        try:
            time.sleep(1)
            
            consent_buttons = [
                "//button[contains(text(), 'Accept')]",
                "//button[contains(text(), 'Agree')]",
                "//button[contains(text(), 'OK')]",
                "//button[contains(text(), 'I agree')]",
                "//button[contains(@class, 'consent')]",
                "//div[contains(@class, 'popup')]//button",
                "//div[contains(@class, 'modal')]//button"
            ]
            
            for selector in consent_buttons:
                try:
                    buttons = self.driver.find_elements(By.XPATH, selector)
                    if buttons:
                        for button in buttons:
                            if button.is_displayed():
                                button.click()
                                print("Clicked popup button")
                                time.sleep(1)
                                break
                except:
                    continue
                    
        except Exception as e:
            print(f"Error in popup handling: {e}")

    def capture_current_captcha(self):
        """Capture the current CAPTCHA image"""
        try:
            # Find the CAPTCHA image
            captcha_img = self.driver.find_element(By.XPATH, "/html/body/div[5]/div[2]/form/div/div/img")
            
            # Take screenshot of the captcha
            captcha_img.screenshot("current_captcha.png")
            
            # Save as a sample
            self.sample_count += 1
            shutil.copy("current_captcha.png", f"captcha_db/samples/{self.sample_count}.png")
            
            print(f"\n--- CAPTCHA #{self.sample_count} ---")
            return True
        except Exception as e:
            print(f"Error capturing CAPTCHA: {e}")
            return False

    def is_refresh_button(self, x, y, w, h, img_width, img_height):
        """Check if a contour is likely the refresh button in the top right"""
        # The refresh button is typically in the top right corner
        # We'll check if the contour is in the top right quadrant of the image
        if x > img_width * 0.7 and y < img_height * 0.3 and w < img_width * 0.2 and h < img_height * 0.2:
            return True
        return False

    def try_multiple_thresholds(self):
        """Try multiple threshold values and show all results"""
        # Load the image
        img = cv2.imread("current_captcha.png")
        if img is None:
            print("Error: Could not load CAPTCHA image")
            return None
        
        # Get image dimensions
        img_height, img_width = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Try different thresholds
        results = []
        
        for threshold in self.thresholds:
            # Apply binary thresholding
            _, binary = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by size and ignore the refresh button
            valid_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter criteria:
                # 1. Not too small (noise)
                # 2. Not the refresh button
                if w * h > 30 and not self.is_refresh_button(x, y, w, h, img_width, img_height):
                    valid_contours.append((x, y, w, h))
            
            # Sort by x-coordinate (left to right)
            valid_contours.sort(key=lambda c: c[0])
            
            # Create visualization
            vis_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
            
            # Draw the contours on the visualization image
            for i, (x, y, w, h) in enumerate(valid_contours):
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.putText(vis_img, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Extract character images
            char_images = []
            for x, y, w, h in valid_contours:
                x_pad, y_pad = 2, 2
                x1, y1 = max(0, x - x_pad), max(0, y - y_pad)
                x2, y2 = min(binary.shape[1], x + w + x_pad), min(binary.shape[0], y + h + y_pad)
                char_img = binary[y1:y2, x1:x2]
                char_images.append(char_img)
            
            # Try to recognize the text using our database
            word_guess, confidence = self.recognize_text(char_images)
            
            results.append({
                'threshold': threshold,
                'binary': binary,
                'visualization': vis_img,
                'char_count': len(valid_contours),
                'char_images': char_images,
                'contours': valid_contours,
                'word_guess': word_guess,
                'confidence': confidence
            })
        
        # Show all results
        self.show_threshold_grid(results)
        
        return results

    def recognize_text(self, char_images):
        """Try to recognize a word from character images using our database"""
        # If database is empty, we can't recognize anything
        if not self.char_db:
            return "", 0
        
        # Recognize each character
        recognized_text = ""
        confidence_scores = []
        
        for char_img in char_images:
            # Resize to standard size
            char_img = cv2.resize(char_img, (20, 30))
            
            # Find the best match in our database
            best_match = None
            best_score = -1
            
            for letter, templates in self.char_db.items():
                for template in templates:
                    # Calculate similarity
                    score = self.calculate_similarity(char_img, template)
                    
                    if score > best_score:
                        best_score = score
                        best_match = letter
            
            # If we found a decent match
            if best_match and best_score > 0.5:
                recognized_text += best_match
                confidence_scores.append(best_score)
            else:
                recognized_text += "?"
                confidence_scores.append(0)
        
        # Calculate overall confidence
        avg_confidence = 0
        if confidence_scores:
            avg_confidence = (sum(confidence_scores) / len(confidence_scores)) * 100
        
        return recognized_text, avg_confidence

    def show_threshold_grid(self, results):
        """Show a grid of all threshold results"""
        # Create figure (make it bigger to fit all 30 thresholds)
        fig = plt.figure(figsize=(20, 40))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # Create a grid for all thresholds (5 columns, multiple rows)
        cols = 5
        rows = (len(results) + cols - 1) // cols
        
        for i, result in enumerate(results):
            plt.subplot(rows, cols, i+1)
            plt.imshow(cv2.cvtColor(result['visualization'], cv2.COLOR_BGR2RGB))
            plt.title(f"#{i+1}: T={result['threshold']} - {result['char_count']} chars")
            plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig("threshold_grid.png")
        plt.close()
        
        print(f"Generated {len(results)} different segmentations. See 'threshold_grid.png'")

    def fast_train(self):
        """Fast training process with multiple threshold options"""
        try:
            # Make sure we're initialized
            if not self.initialized:
                self.initialize()
            
            # Capture current CAPTCHA
            if not self.capture_current_captcha():
                return False
            
            # Try multiple thresholds
            results = self.try_multiple_thresholds()
            if not results:
                return False
            
            # Display threshold options with guesses
            print("\nChoose the option with the best character segmentation:")
            print("Option | Threshold | Characters | Guess | Confidence")
            print("-" * 60)
            
            for i, result in enumerate(results):
                guess_text = result['word_guess'] if result['word_guess'] else "?"
                print(f"{i+1:2d} | {result['threshold']:3d} | {result['char_count']:2d} chars | {guess_text:10s} | {result['confidence']:.1f}%")
            
            print("\nEnter 1-30 for the corresponding threshold, or 's' to skip: ")
            choice = input()
            
            if choice.lower() == 's':
                print("Skipping this CAPTCHA")
                self.skipped_count += 1
                return False
            
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(results):
                    selected = results[choice_idx]
                    print(f"Selected option #{int(choice)}: threshold {selected['threshold']} with {selected['char_count']} characters")
                    
                    if selected['word_guess']:
                        print(f"Guess for this option: '{selected['word_guess']}' ({selected['confidence']:.1f}%)")
                    
                    # Ask for the correct text
                    print("Enter the correct CAPTCHA text: ")
                    actual_text = input().strip().lower()
                    
                    if not actual_text:
                        print("No text entered, skipping")
                        self.skipped_count += 1
                        return False
                    
                    # Check if segment count matches text length
                    if selected['char_count'] != len(actual_text):
                        print(f"Warning: {selected['char_count']} segments but {len(actual_text)} characters")
                        print("Enter 'y' to continue anyway, any other key to skip: ")
                        if input().lower() != 'y':
                            print("Skipping this CAPTCHA")
                            self.skipped_count += 1
                            return False
                    
                    # Calculate accuracy if there was a guess
                    if selected['word_guess']:
                        accuracy = self.calculate_accuracy(selected['word_guess'], actual_text)
                        print(f"Guess accuracy: {accuracy:.1f}%")
                    
                    # Train on this CAPTCHA
                    self.train_on_selected(selected, actual_text)
                    self.trained_count += 1
                    return True
                else:
                    print("Invalid selection, please enter a number between 1 and 30")
                    return False
            except ValueError:
                print("Invalid input, please enter a number or 's'")
                return False
            
        except Exception as e:
            print(f"Error in fast training: {e}")
            return False

    def calculate_accuracy(self, guess, actual):
        """Calculate accuracy between guess and actual text"""
        if not actual:
            return 0.0
            
        # Count matching characters
        correct = 0
        for i in range(min(len(guess), len(actual))):
            if guess[i] == actual[i]:
                correct += 1
                
        # Calculate accuracy percentage
        return (correct / max(len(guess), len(actual))) * 100

    def train_on_selected(self, selected, actual_text):
        """Train on the selected threshold result"""
        char_images = selected['char_images']
        
        # Associate each character image with the corresponding letter
        max_chars = min(len(char_images), len(actual_text))
        
        # Keep track of new characters added
        new_chars = set()
        
        for i in range(max_chars):
            char_img = char_images[i]
            char_letter = actual_text[i]
            
            # Save character image with its actual letter
            char_filename = f"captcha_db/chars/{char_letter}_{len(self.char_db.get(char_letter, []))}.png"
            cv2.imwrite(char_filename, char_img)
            
            # Resize to a standard size for better comparison
            char_img = cv2.resize(char_img, (20, 30))
            
            # Add to database
            if char_letter not in self.char_db:
                self.char_db[char_letter] = []
                new_chars.add(char_letter)
            
            # Add to database (up to 10 examples per letter for diversity)
            if len(self.char_db[char_letter]) < 10:
                self.char_db[char_letter].append(char_img)
                
        # Save the updated database
        with open(self.db_file, "wb") as f:
            pickle.dump(self.char_db, f)
        
        if new_chars:
            print(f"Added new characters to database: {', '.join(sorted(new_chars))}")
        
        print(f"Database now has {len(self.char_db)} unique characters with {sum(len(v) for v in self.char_db.values())} total templates")

    def calculate_similarity(self, img1, img2):
        """Calculate similarity between two character images"""
        # Template matching using correlation
        try:
            if img1.shape != img2.shape:
                img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
            
            # Calculate absolute difference
            diff = cv2.absdiff(img1, img2)
            
            # Calculate similarity (inverse of difference)
            similarity = 1.0 - (np.sum(diff) / (img1.shape[0] * img1.shape[1] * 255))
            
            return similarity
        except:
            return 0

    def reload_captcha(self):
        """Reload the CAPTCHA by clicking the refresh button or refreshing the page"""
        try:
            # Find and click the refresh button if available
            try:
                refresh_button = self.driver.find_element(By.XPATH, "//button[@type='button']")
                refresh_button.click()
                time.sleep(1)
            except:
                # If no refresh button, reload the page
                self.driver.refresh()
                time.sleep(3)
                self.handle_popups()
            
            # Wait for captcha to appear
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/div[5]/div[2]/form/div/div/img"))
            )
            
            print("CAPTCHA reloaded")
            return True
        except Exception as e:
            print(f"Error reloading CAPTCHA: {e}")
            return False
    
    def show_database_stats(self):
        """Show statistics about the current database"""
        if not self.char_db:
            print("Database is empty")
            return
            
        print("\n=== Database Statistics ===")
        print(f"Total unique characters: {len(self.char_db)}")
        print(f"Total templates: {sum(len(v) for v in self.char_db.values())}")
        
        print("\nCharacters by number of templates:")
        for char, templates in sorted(self.char_db.items()):
            print(f"  '{char}': {len(templates)} templates")
            
        print("\nCharacters missing from database:")
        all_chars = set("abcdefghijklmnopqrstuvwxyz")
        missing = all_chars - set(self.char_db.keys())
        if missing:
            print(f"  Missing: {', '.join(sorted(missing))}")
        else:
            print("  None! All characters are in the database.")

    def reset_database(self):
        """Reset the database if needed"""
        confirm = input("Are you sure you want to reset the database? (y/n): ")
        if confirm.lower() == 'y':
            self.char_db = {}
            with open(self.db_file, "wb") as f:
                pickle.dump(self.char_db, f)
            print("Database reset complete")

    def close(self):
        self.driver.quit()

# Main execution
if __name__ == "__main__":
    trainer = FastCaptchaTrainer()
    
    try:
        print("=== Enhanced CAPTCHA Trainer with Guessing ===")
        print("This system will try 30 different threshold values and guess each CAPTCHA")
        
        # Initialize
        trainer.initialize()
        
        # Ask if they want to reset the database
        reset = input("Do you want to reset the current database? (y/n): ")
        if reset.lower() == 'y':
            trainer.reset_database()
        
        # Training loop
        while True:
            trainer.fast_train()
            
            # Reload CAPTCHA for next round
            trainer.reload_captcha()
            
            # Show stats every 5 CAPTCHAs
            if (trainer.trained_count + trainer.skipped_count) % 5 == 0:
                trainer.show_database_stats()
                
                # Ask to continue
                cont = input("\nContinue training? (y/n): ")
                if cont.lower() != 'y':
                    break
        
        print("\nTraining complete!")
        print(f"CAPTCHAs trained: {trainer.trained_count}")
        print(f"CAPTCHAs skipped: {trainer.skipped_count}")
        trainer.show_database_stats()
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        print("Closing browser and saving database...")
        trainer.close()
        print("Training session complete!")