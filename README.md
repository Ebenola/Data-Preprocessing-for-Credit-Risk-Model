**<h1>Credit Risk Model: Estimating Probability of Default</h1>**

**<h2>Overview</h2>**

This project aims to create a credit risk model that estimates the probability of default for every personal account. The raw dataset was preprocessed to obtain a clean and usable dataset for the model.

**<h2>Steps</h2>**
<h3>1. Currency Conversion</h3>
The raw dataset is currently in Dollars. I converted it to Euros for consistency.
All monetary values was appropriately adjusted.
<h3>2. Quantifying Categorical Variables</h3>
Every categorical variable needs to be quantified. All text columns were converted into numerical representations.
This step ensures that the model can handle categorical features effectively.
<h3>3. Handling Missing Data</h3>
Missing information is a red flag for this loan application. It was addressed cautiously.
The dataset contains <b>88,005 missing elements</b>.

**Dataset Details**
The data is sourced from a CSV file called loan-data.csv from a financial services company, lendingclub.com
(365 Data Science).

The rows can be referred to as <b>Accounts, Candidates, or Applications</b>.

<pre>import numpy as np

np.set_printoptions(suppress=True, linewidth=100, precision=2)

raw_data_np = np.genfromtxt("loan-data.csv", delimiter=';', skip_header=1, autostrip=True)

missing_count = np.isnan(raw_data_np).sum()

print(f"Missing elements in the dataset: {missing_count}")

temporary_fill = np.nanmax(raw_data_np) + 1

temporary_mean = np.nanmean(raw_data_np, axis = 0)</pre>

**<h3>4. Handling String Columns</h3>**
The observation is that 8 columns in the dataset contain strings. These columns likely represent categorical variables or textual information.

To handle them, I splitted the dataset into two parts: one containing numeric values and the other containing strings.

**<h3>5. Extracting Descriptive Statistics</h3>**
Before splitting, I extracted some descriptive statistics for each column:

<b>Minimum Value:</b> The smallest value in each column.

<b>Mean Value:</b> The average value (excluding missing data).

<b>Maximum Value:</b> The largest value in each column.

<pre>temporary_stats = np.array([np.nanmin(raw_data_np, axis=0), temporary_mean, np.nanmax(raw_data_np, axis=0)])

columns_strings = np.argwhere(np.isnan(temporary_mean)).squeeze()

columns_numeric = np.argwhere(np.isnan(temporary_mean) == False).squeeze()</pre>

**<h3>6. Handling Column Headers</h3>**
Store the headers of each column to keep track of the information stored in each column.
Extract the column headers from the original dataset.

<pre>header_full = np.genfromtxt("loan-data.csv",
                            delimiter=';',
                            skip_footer=raw_data_np.shape[0],
                            autostrip=True,
                            dtype=np.str)

header_strings, header_numeric = header_full[columns_strings], header_full[columns_numeric]</pre>

**<h3>7. Creating Checkpoints</h3>**
Checkpoints are essential to prevent accidental data loss during code development.

Checkpoint was created to store a copy of the dataset and headers.

<pre>def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header=checkpoint_header, data=checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return checkpoint_variable
checkpoint_test = checkpoint("checkpoint_test", header_strings, loan_data_strings)</pre>

**<h3>8. Renaming Columns</h3>**
The first header was made more descriptive by renaming it from 'Issue_d' to 'Issue_date'.

<pre>header_strings[0] = "Issue_date"</pre>

**<h3>9. Handling Date Format</h3>**
The first column ('Issue_date') follows a pattern: all months are represented by 3 alphabets, and all are from the year 2015.

Remove the common -15 suffix from the dates.

<pre>loan_data_strings[:, 0] = np.chararray.strip(loan_data_strings[:, 0], "-15")</pre>

The months in the 'Issue_date' column were converted to integers (1 for January, 2 for February, etc.).

This optimization reduces memory usage and ensures consistency.
<pre>months = np.array(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
for i in range(13):
    loan_data_strings[:, 0] = np.where(loan_data_strings[:, 0] == months[i], i, loan_data_strings[:, 0])</pre>
<pre># Check again to ensure there are no empty spaces
np.unique(loan_data_strings[:, 0])</pre>

**<h3>11. Handling Loan Status</h3>**
I simplified the loan status into two categories: ‘Good’ and ‘Bad’.

Accounts with status like <b>'Current', 'Fully Paid', 'In Grace Period', and 'Issued' were grouped as ‘Good’</b>.

<b>'Charged Off', empty values, and 'Default' were grouped as ‘Bad’.</b>

Short-period latencies like 'Late (16 - 30 days)' was considered ‘Good’, while longer periods (31 - 120 days) as ‘Bad’.

<b>Represent ‘Good’ with 1 and ‘Bad’ with 0.</b>

<pre>status_bad = np.array(['', 'Charged Off', 'Default', 'Late (31-120 days)'])
loan_data_strings[:, 1] = np.where(np.isin(loan_data_strings[:, 1], status_bad), 0, 1)

# Confirm that the numbers 0 and 1 have been properly assigned
np.unique(loan_data_strings[:, 1])
</pre>

**<h3>12. Handling Loan Term</h3>**
Strip the common 'months' suffix from the 'term' column.

Additionally, the header name was changed to be more descriptive.

<pre># Strip the 'months' suffix
loan_data_strings[:, 2] = np.chararray.strip(loan_data_strings[:, 2], " months")

# Change the header name
header_strings[2] = "term_months"</pre>

60 was assigned to the missing values in the 'term_months' column. Since 60 is the highest value in the column, assume the worst-case scenario for missing values.

<pre>loan_data_strings[:, 2] = np.where(loan_data_strings[:, 2] == '', '60', loan_data_strings[:, 2])</pre>

**<h3>14. Handling Sub-Grade</h3>
The 'sub_grade' column appears related to the 'grade'.
For every element in 'grade', there are five elements in 'sub_grade'. Therefore, information in the 'grade' column can be obtained from the 'sub_grade' column.
Assign the worst sub-grade (e.g., 'G5') to missing values.
<pre>
# Assigning the worst sub-grade to missing values
for i in np.unique(loan_data_strings[:, 3])[1:]:
    loan_data_strings[:, 4] = np.where((loan_data_strings[:, 4] == '') & (loan_data_strings[:, 3] == i),
                                       i + '5', loan_data_strings[:, 4])

# Confirm that the output remains the same (There are still missing data)
np.unique(loan_data_strings[:, 4], return_counts=True)</pre>

**<h3>15. Creating a New Category</h3>**
A new category lower than 'G5' was created and assigned to the missing values.

There is no longer need for the 'grade' column (all info can be found in 'sub_grade'), hence, delete the 'grade' column.
<pre>
# Create a new category (lower than G5) for missing values
loan_data_strings[:, 4] = np.where(loan_data_strings[:, 4] == '', "H1", loan_data_strings[:, 4])

# Delete the 'grade' column
loan_data_strings = np.delete(loan_data_strings, 3, axis=1)

# Confirm that the 'grade' column is now the fourth column
loan_data_strings[:, 3]</pre>

**Update the Header**
Header was updated to reflect the changes made.

<pre># Effecting the change in the header
header_strings = np.delete(header_strings, 3)</pre>

<pre># Assign numbers to each sub-grade from A1 to H1
keys = list(np.unique(loan_data_strings[:, 3]))
values = list(range(1, np.unique(loan_data_strings[:, 3]).shape[0] + 1))
dict_sub_grade = dict(zip(keys, values))

# Apply the mapping
for i in np.unique(loan_data_strings[:, 3]):
    loan_data_strings[:, 3] = np.where(loan_data_strings[:, 3] == i, dict_sub_grade[i], loan_data_strings[:, 3])

# Confirm the unique sub-grades
np.unique(loan_data_strings[:, 3])</pre>

**<h3>16. Handling Loan Verification Status</h3>**
Assumption: the missing value in the 'verification_status' column is equivalent to ‘Not Verified’.
Assign numeric values (0 or 1) to represent verified or not verified.
<pre>
# Assume missing value is equivalent to 'Not Verified'
loan_data_strings[:, 4] = np.where((loan_data_strings[:, 4] == '') | (loan_data_strings[:, 4] == 'Not Verified'), 0, 1)</pre>

**<h3>17. Handling Loan ID</h3>**
The URL address is identical for all loans except for the loan ID.
The common URL prefix was stripped, leaving only the numeric loan ID stored as strings.
Confirm that the 'id' column in the full header matches the extracted loan IDs.

<pre># Strip the common URL prefix from the loan ID
loan_data_strings[:, 5] = np.chararray.strip(loan_data_strings[:, 5], 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=')

# Confirm that the loan IDs match
np.array_equal(loan_data_strings[:, 5].astype(dtype=np.int32), loan_data_numeric[:, 0].astype(dtype=np.int32))

# Delete the URL column and update the header
loan_data_strings = np.delete(loan_data_strings, 5, axis=1)
header_strings = np.delete(header_strings, 5)
</pre>

**<h3>18. Handling State Addresses</h3>**
The last column name was changed to state_address.
The state_address column contains USA states’ abbreviations and some missing values.
<pre>
# Change the name of the last column to be more relatable
header_strings[5] = "state_address"

# Check the unique values in the state_address column
np.unique(loan_data_strings[:, 5])</pre>

Group the states based on common characteristics like geographic location.

This grouping will allow effective handling of outlier and ensure robust coefficients.

A unique value was assigned to each state, including the missing values (assigned as 0).

<pre># Assign zero to missing values
loan_data_strings[:, 5] = np.where(loan_data_strings[:, 5] == '', 0, loan_data_strings[:, 5])

# Group the states into West, South, Midwest, and East
states_west = np.array(['ID', 'MT', 'NV', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA', 'AZ', 'CO', 'NM'])
states_south = np.array(['OK', 'AR', 'LA', 'MS', 'AL', 'TN', 'KY', 'GA', 'SC', 'NC'])
states_midwest = np.array(['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH'])
states_east = np.array(['PA', 'NY', 'VT', 'NH', 'ME', 'MA', 'RI', 'CT', 'NJ'])
</pre>

Assign: 1 for states in the West;
2 for states in the South;
3 for states in the Midwest;
4 for states in the East

<pre># Assign numeric values to states based on geographic location
loan_data_strings[:, 5] = np.where(np.isin(loan_data_strings[:, 5], states_west), 1, loan_data_strings[:, 5])
loan_data_strings[:, 5] = np.where(np.isin(loan_data_strings[:, 5], states_south), 2, loan_data_strings[:, 5])
loan_data_strings[:, 5] = np.where(np.isin(loan_data_strings[:, 5], states_midwest), 3, loan_data_strings[:, 5])
loan_data_strings[:, 5] = np.where(np.isin(loan_data_strings[:, 5], states_east), 4, loan_data_strings[:, 5])

# Confirm that the numbers have been assigned appropriately
np.unique(loan_data_strings[:, 5])
</pre>

**<h3>19. Converting to Numeric Format</h3>**
All string data have been converted into numeric values stored as text.
Now convert the data into a format that numpy recognizes as numeric.

<pre># Convert the data into integers
loan_data_strings = loan_data_strings.astype(np.int)</pre>

<b> Create Checkpoints for all the changes </b>
<pre>checkpoint_strings = checkpoint("checkpoint-strings", header_strings, loan_data_strings)
checkpoint_strings["header"]
checkpoint_strings["data"]</pre>

<pre># Confirm that the values in the checkpoint are equal to the processed data
np.array_equal(checkpoint_strings['data'], loan_data_strings)</pre>

**<h3>20. Cleaning and Preprocessing Numeric Data</h3>**
Cleaning and preprocessing the numeric data.
<pre>
# Check if there are missing values in the numeric array
np.isnan(loan_data_numeric).sum()

# Confirm the headers of the numeric values
header_numeric</pre>

Set the temporary_fill values to minimum in the funded amount column

<pre>loan_data_numeric[:,2] = np.where(loan_data_numeric[:,2] == temporary_fill, temporary_stats[0, columns_numeric[2]],
                                  loan_data_numeric[:,2])
loan_data_numeric[:,2]</pre>

Set the max values for loan_amount, interest_rate, installment, and total_payment

<pre># This will set temporary_fill values to maximum temporary stats

for i in [1,3,4,5]:
    loan_data_numeric[:,i] = np.where(loan_data_numeric[:,i] == temporary_fill, temporary_stats[2, columns_numeric[i]], 
                                     loan_data_numeric[:,i])
loan_data_numeric</pre>

**<h3>21. Working on the Exchange Rate</h3>**
Exchange rate data was imported from the file “EUR-USD.csv.” Initially, the data displayed NaN in the first row, which likely corresponds to the column names. 
Set the data type to strings.

<pre># Import exchange rate converter file
EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter=',', autostrip=True, dtype=str)</pre>

The columns in the EUR_USD array represent ‘Open,’ ‘High,’ ‘Close,’ and ‘Volume.’ The interest here is the adjusted closing price, which is stored in the fourth column (index 3):

<pre># Extract the adjusted closing price (fourth column)
EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter=',', autostrip=True, skip_header=1, usecols=3)
</pre>

Setting exchange rates for different months

<pre>exchange_rate = loan_data_strings[:, 0]

for i in range(1, 13):
    exchange_rate = np.where(exchange_rate == i, EUR_USD[i - 1], exchange_rate)

# For the missing month 0, the exchange rate will be the mean
exchange_rate = np.where(exchange_rate == 0, np.mean(EUR_USD), exchange_rate)
</pre>

Add this new array of exchange rates to the numeric dataset. Check if the new variable and loan_data_numeric have compatible shapes

<pre>exchange_rate.shape
loan_data_numeric.shape
</pre>

The shape of exchange_rate is 1-dimensional and the numeric dataset is 2-dimensional, hence reshape exchange_rate to match the dimensions

<pre>exchange_rate = np.reshape(exchange_rate, (10000, 1))
</pre>

Horizontally stack the exchange rate column with the existing numeric dataset and update the header

<pre>np.hstack((loan_data_numeric, exchange_rate))
loan_data_numeric = np.hstack((loan_data_numeric, exchange_rate))

# Add the new exchange rate column to the header
header_numeric = np.concatenate((header_numeric, np.array(['exchange_rate'])))
</pre>

Four columns in the dataset contain values in US dollars: loan_amnt, funded_amnt, installment, and total_pymnt. 
Convert these amounts to Euro using the exchange rate calculated earlier

The value in each of the dollar column was divided by the exchange rate to get the equivalent amount in Euro. The exchange rate array was reshaped to match the dimensions of our dataset.

Updating the Dataset: add the new Euro columns to the numeric dataset and update the header accordingly.

<pre># Import exchange rate converter file
EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter=',', autostrip=True, dtype=str)

# Extract the adjusted closing price (fourth column)
EUR_USD = np.genfromtxt("EUR-USD.csv", delimiter=',', autostrip=True, skip_header=1, usecols=3)
</pre>

<pre># Divide dollar amounts by the exchange rate to get Euro values
for i in columns_dollar:
    loan_data_numeric = np.hstack((loan_data_numeric, np.reshape(loan_data_numeric[:, i] / loan_data_numeric[:, 6], (10000, 1))))
</pre>
<pre># Create additional header for Euro columns
header_additional = np.array([column_name + '_EUR' for column_name in header_numeric[columns_dollar]])

# Add the additional header to the existing header
header_numeric = np.concatenate((header_numeric, header_additional))

# Update the default column names to indicate they are in USD
header_numeric[columns_dollar] = np.array([column_name + '_USD' for column_name in header_numeric[columns_dollar]])

# Rearrange the columns to group USD and Euro columns together
columns_index_order = [0, 1, 7, 2, 8, 3, 4, 9, 5, 10, 6]
header_numeric = header_numeric[columns_index_order]
loan_data_numeric = loan_data_numeric[:, columns_index_order]
</pre>
<pre># Transform interest rate values
loan_data_numeric[:, 5] = loan_data_numeric[:, 5] / 100
</pre>

**<h3>22. Creating a Checkpoint for Preprocessed Numeric Values</h3>**
Checkpoint saved the preprocessed numeric values. This will enable easier revert to this state if needed

<pre># Create a checkpoint for all the preprocessed numeric values
checkpoint_numeric = checkpoint("checkpoint-numeric", header_numeric, loan_data_numeric)
</pre>

**<h3>23. Combining Numeric and String Data</h3>**
<pre># Combine the two arrays
loan_data = np.hstack((checkpoint_numeric['data'], checkpoint_strings['data']))</pre>

**<h3>24. Confirming Data Completeness</h3>**
<pre># Confirm there are no missing values
np.isnan(loan_data).sum()
</pre>

<pre># Sort the data by loan ID
loan_data = loan_data[np.argsort(loan_data[:, 0])]
</pre>

<pre># Stack the header on top of the sorted data
loan_data = np.vstack((header_full, loan_data))
</pre>

**<h3>25. Saving the Preprocessed Dataset</h3>**
<pre># Save the preprocessed dataset
np.savetxt("loan-data-preprocessed.csv", loan_data, fmt="%s", delimiter=',')
</pre>

<b>The dataset is now cleaned, preprocessed, and ready for further analysis. 
The dataset could be used to prepare a credit risk model (CRM) for estimating the probability of default.</b>
