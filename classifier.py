#TODO: Complete data cleaning module
import pandas as pd

#TODO: Loading fake and real data sets in csv format
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

#TODO: Set column to identify fake news
fake["fake_news"] = 1
true["fake_news"] = 0

#TODO: Examine the unique article subjects
fake["subject"].unique()
true["subject"].unique()

#TODO: Verification that data loaded properly
print(fake.head())
print(true.head())
#NOTE: Verification done correctly

#TODO: Extracting columns to remove overfitting
true_text = true["text"]

#leaves remaining text after hyphen unchanged
true_text = true_text.str.extractall(r"^.* - (?P<text>.*)")
true_text = true_text.droplevel(1)
true = true.assign(text=true_text["text"]) #switcharoo on columns

# print(true.head())

#TODO: Combining, Concatenating, and Saving Data
#NOTE: df stands for data frame (like a data structure)
df = pd.concat([fake, true], axis = 0) #axis=1 line the column up

#TODO: Dropping other unnecessary columns
df = df.drop(["subject", "date", "title"], axis = 1)

#TODO: Verification that data loaded properly
print(df.head())
print(df.tail())
#NOTE: Verification done correctly

#TODO: Check for missing records
df.dropna(axis = 0)
print(df.info())

#TODO: Save cleaned CSV for other modeling
clean_text = df.to_csv("cleaned_news.csv", index = False)
