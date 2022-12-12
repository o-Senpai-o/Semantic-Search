
import dask.bag as db
import json
from datetime import datetime
import time

def v1_date(row):
    """
    For each row in the dask bag, 
    find the date of the first version of the paper 
    and add it to the row as a new column
    Args:
      row: a row of the dask bag
    Returns:
      A row of the dask bag with added "unix_time" column
    """
    
    versions = row["versions"]

    date = None
    for version in versions:
        if version["version"] == "v1":
            date = datetime.strptime(version["created"], "%a, %d %b %Y %H:%M:%S %Z")
            date = int(time.mktime(date.timetuple()))

    row["unix_time"] = date

    return row


def text_col(row):
    """
    It takes a row of a dataframe, adds a new column called 'text' 
    that is the concatenation of the 'title' and 'abstract' columns
    Args:
      row: the row of the dataframe
    Returns:
      A row with the text column added.
    """

    row["text"] = row["title"] + "[SEP]" + row["abstract"]
    return row


def filters(row):
    """
    For each row in the dask bag, only keep the row if it meets the filter criteria
    
    Args:
      row: the row of the dataframe
    Returns:
      Boolean mask
    """
    
    return ((len(row["id"])<16) and 
            (len(row["categories"])<200) and
            (len(row["title"])<4096) and
            (len(row["abstract"])<65535) and
            ("cs." in row["categories"]) # Keep only CS papers
           )



