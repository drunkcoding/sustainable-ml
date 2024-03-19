import time
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt

URL = "https://www.techpowerup.com/gpu-specs/?eol=Active"

def get_choices_url(url):
    print(url)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    # get id of the dropdown
    dropdown = soup.find_all("select")
    dropdown_ids = [dropdown["id"] for dropdown in dropdown if "id" in dropdown.attrs]

    # get all choices from each dropdown id
    choices = {}
    for dropdown_id in dropdown_ids:
        choices[dropdown_id] = [option.text for option in soup.find(id=dropdown_id).find_all("option")]
    # exclude the first choice which is "All"
    choices = {dropdown_id: choices[dropdown_id][1:] for dropdown_id in choices}
    return choices

def get_processors_table(url):
    print(url)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    # get the table
    tables = soup.find_all("table")
    if len(tables) == 0:
        return None
    
    tables = [table for table in tables if "class" in table.attrs and "processors" in table["class"]][0]

    # get the headers
    headers = tables.find_all("th")
    headers = [header.text for header in headers]
    headers = headers[1:]  # exclude the first header which is "Model"

    # get the rows
    rows = tables.find_all("tr")
    rows = [row.find_all("td") for row in rows]
    rows = [[cell.text for cell in row] for row in rows]

    # remove \n from the cells
    rows = [[cell.replace("\n", "") for cell in row] for row in rows]

    # create the dataframe
    df = pd.DataFrame(rows, columns=headers)
    return df

choices = get_choices_url(URL)
# eol = end of list. Select the active choice
print(choices)
time.sleep(5)

# Manufacturer -> mfgr
for mfgr in choices["mfgr"]:
    mfgr = mfgr.split(" ")[0]
    print(mfgr)
    mfgr_choices = get_choices_url(f"{URL}&mfgr={mfgr}")
    print(mfgr_choices)

    for released in mfgr_choices["released"]:
        released = released.split(" ")[0]
        print(released)
        df = get_processors_table(f"{URL}&mfgr={mfgr}&released={released}")
        print(df)
        # exit()

        time.sleep(5)
# GPU Series -> series