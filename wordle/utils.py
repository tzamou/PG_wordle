from selenium.webdriver import Chrome,ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time,re

class Web:
    def __init__(self,invisiable=True):
        if invisiable==True:
            options = ChromeOptions()
            options.add_argument("--headless")
            self.driver = Chrome(chrome_options=options, executable_path='../web/chromedriver')
            self.driver.get('https://www.nytimes.com/games/wordle/index.html')
        elif invisiable==False:
            self.driver = Chrome('../web/chromedriver')
            self.driver.get('https://www.nytimes.com/games/wordle/index.html')
        time.sleep(0.2)
        ActionChains(self.driver).move_by_offset(200, 100).click().perform()
    def closeweb(self):
        self.driver.quit()

    def answer(self,ans):
        for alphabet in ans:
            #time.sleep(0.1)N
            ActionChains(self.driver).move_by_offset(0, 0).key_down(alphabet).perform()

        ActionChains(self.driver).move_by_offset(0, 0).key_down(u'\ue007').perform()

    def result(self):
        time.sleep(0.5)
        inner_texts = [my_elem.get_attribute("outerHTML") for my_elem in self.driver.execute_script(f"return document.querySelector('game-app').shadowRoot.querySelector('game-row').shadowRoot.querySelectorAll('game-tile[letter]')")]
        lst = re.findall(r'evaluation="[a-z]+"', str(inner_texts))
        lst = re.findall(r'"[a-z]+"', str(lst))
        result=[0,0,0,0,0]
        for i in range(5):
            if lst[i]=='"correct"':
                result[i]=1
            elif lst[i]=='"present"':
                result[i]=2
            elif lst[i]=='"absent"':
                result[i]=3
        return result

if __name__=='__main__':
    web=Web(invisiable=False)
    web.answer('swill')
    web.result()
