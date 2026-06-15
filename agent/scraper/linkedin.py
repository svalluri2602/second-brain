import os
import asyncio
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from dataclasses import dataclass
import sqlite3



load_dotenv()

@dataclass 
class LinkedInProfile:

    url: str
    name: str
    headline: str
    about: str
    experience: str = ""
    posts: list = None
    scraped_at: str = ""



class LinkedInScraper:
    
    def __init__(self):
        self.email = os.getenv("LINKEDIN_EMAIL")
        self.password = os.getenv("LINKEDIN_PWD")
    
    async def login(self, page):
        # your login code goes here
        await page.goto("https://www.linkedin.com/login")
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(2000)
        await page.wait_for_selector('[autocomplete="username webauthn"]')
        await page.locator('[autocomplete="username webauthn"]').fill(self.email)
        await page.wait_for_timeout(1000)
        await page.locator('input[type="password"]').nth(1).fill(self.password, force=True)
        await page.wait_for_timeout(500)
        await page.keyboard.press("Enter")
        await page.wait_for_url("**/feed/**", timeout=15000)
        pass  
    
    async def scrape_profile(self, page, url):
        # scrape a profile given a URL
        await page.goto(url)
        await page.wait_for_timeout(3000)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(3000)
        content = await page.content()
        print(content[:3000])
        name = await page.locator('h2').nth(1).inner_text()
        await page.evaluate("window.scrollTo(0, 1000)")
        await page.wait_for_timeout(2000)
        print(f"Name: {name}")
        first_p = await page.locator('section[aria-label="Primary content"] p').first.inner_text()
        second_p = await page.locator('section[aria-label="Primary content"] p').nth(1).inner_text()

        if len(first_p) < 12:
             headline = second_p
        else:
             headline = first_p
    
        print(f"Headline: {headline}")

        await page.evaluate("window.scrollTo(0, 1000)")
        await page.wait_for_timeout(1000)


        try:
            await page.evaluate("window.scrollTo(0, 1500)")
            await page.wait_for_timeout(1000)
            about = await page.locator('[data-testid="expandable-text-box"]').first.inner_text()
            print(f"ABOUT: {about}")
        
        except:
            about = ""
            print(f"ABOUT: {about}")

        
        #await page.get_by_text("Show all").scroll_into_view_if_needed()
        await page.wait_for_timeout(3000)
        await page.goto(url + "recent-activity/all/")
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(2000)
       # await page.get_by_role("link", name="Show all posts", ).first.click()

       #
        for _ in range(5):
            await page.evaluate("window.scrollBy(0,800)")
            await page.wait_for_timeout(200)

        
        await page.wait_for_timeout(3000)

       
        print(f"Current URL: {page.url}")


        posts_elements = await page.locator('div.update-components-text span[dir="ltr"]').all()
        print(f"Total post elements: {len(posts_elements)}")

        posts = []
        for el in posts_elements[:10]:
            text = await el.inner_text()
            if text and len(text.strip()) > 20:
                posts.append(text.strip())
                print(f"POST: {text.strip()[:150]}")

        print(f"Total posts found: {len(posts)}")
     

        profile = LinkedInProfile(url = url, name = name, headline = headline, about = about, experience= "" , posts = posts, scraped_at= "")

        self._save_to_db(profile)

        

      

           




#scrape posts



        
        #await page.locator('a[aria-label="Show all"]').click()

    pass
    
    async def search_people(self, query, limit):
        # search linkedin and return list of profiles
        pass
    
    def _save_to_db(self, profile):
        # save LinkedInProfile to bux.db
        import json
        conn = sqlite3.connect("bux.db");
        cursor = conn.cursor()
        cursor.execute(f"""
         INSERT INTO linkedin VALUES
         (NULL, ?, ?, ?, ?, ?, ?, ?)  """,
         (profile.name, profile.url, profile.headline, profile.about, profile.experience, json.dumps(profile.posts), profile.scraped_at))
        conn.commit()



        pass





async def main():

   

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        
        scraper = LinkedInScraper()
        
        # Step 1: login and save session

        if os.path.exists("auth.json"):
            context = await browser.new_context(storage_state = "auth.json")
            page = await context.new_page()

        else:
            context = await browser.new_context()
            page = await context.new_page()
            await scraper.login(page)
            await context.storage_state(path="auth.json")
        
           

      
        
        # Step 2: load session and scrape
    
        await scraper.scrape_profile(page, "https://www.linkedin.com/in/praneel-joshi-b363101b3/")
        await browser.close()



    

asyncio.run(main())




