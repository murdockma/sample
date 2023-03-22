import pandas as pd
import numpy as np

class WeeklyMetrics:
    """
    A class for calculating weekly metrics for the call center (Automates spreadhseet creation).

    Attributes:
    call_center_csv (str): The file path for the call center data.
    dials_csv (str): The file path for the dials data.
    contacts_csv (str): The file path for the contacts data.
    five9_csv (str): The file path for the Five9 data.
    paylocity_csv (str): The file path for the Paylocity data.
    start_date (str): The start date for the week to be analyzed in "YYYY-MM-DD" format.
    end_date (str): The end date for the week to be analyzed in "YYYY-MM-DD" format.

    Methods:
    get_sets(): Extracts information about sets from the call center dataframe and returns a DataFrame with columns for the agent name and their number of sets.
    get_contacts(): Extracts information about contacts from the contacts dataframe and returns a DataFrame with columns for the agent name and their number of contacts and sets.
    get_dials(): Extracts information about dials from the dials dataframe and returns a DataFrame with columns for the agent name and their number of dials along with other calculated values.
    calculate_five9_calling_hours(): Calculates the total time each agent spent in a "Ready" state, an "On Call" state, and on the phone ("Five9 calling hours") based on the input data frame of Five9 call center agent data and returns a DataFrame with columns for the agent name and their total Five9 calling hours.
    """
    
    def __init__(self, call_center_csv, dials_csv, contacts_csv, five9_csv, paylocity_csv, start_date, end_date):
        self.call_center_df = pd.read_csv(call_center_csv)
        self.dials_df = pd.read_csv(dials_csv)
        self.contacts_df = pd.read_csv(contacts_csv)
        self.five9_df = pd.read_csv(five9_csv)
        self.paylocity_df = pd.read_csv(paylocity_csv, header=None)
        self.start_date = start_date
        self.end_date = end_date
        
    def get_sets(self):
        """
        Extracts information about sets from the call center dataframe.
        
        Returns:
            A DataFrame with columns for the agent name and their number of sets.
        """
        
        # Normalize the time in the call center dataframe to just the date
        self.call_center_df['Date Only'] = pd.to_datetime(self.call_center_df['Date/Time']).dt.normalize()

        # Create a boolean mask to select the rows that fall within the given date range
        date_mask = (self.call_center_df['Date Only'] >= self.start_date) & (self.call_center_df['Date Only'] <= self.end_date)
        week_df = self.call_center_df.loc[date_mask]

        # Select only the rows that represent sets where the bonus is not equal to $0.00
        set_comp = week_df[week_df['WT/SA Bonus'] != '$0.00']

        # Count the number of sets for each agent
        sets_df = set_comp['BCI Caller'].apply(lambda x: x.lower()).value_counts().reset_index()
        sets_df = sets_df.rename(columns={'index':'AGENT', 'BCI Caller':'Sets'})
        sets_df['AGENT'] = sets_df['AGENT'].apply(lambda x: x.lower())
        
        return sets_df
        
    def get_contacts(self):
        """
        Extracts information about contacts from the contacts dataframe.
        
        Returns:
            A DataFrame with columns for the agent name and their number of contacts and sets.
        """
        
        # Extract the agent username from the email address
        self.contacts_df['AGENT'] = self.contacts_df['AGENT'].apply(lambda x: x.split("@")[0])

        # Sum the columns that represent contacts for each agent
        self.contacts_df['total contacts'] = self.contacts_df.iloc[:,4:].sum(axis=1)

        # Select only the columns we need
        contact_df_final = self.contacts_df[['AGENT','total contacts']]
        contact_df_final = contact_df_final.rename(columns = {'total contacts':'Contacts'})
        
        # Merge the sets dataframe with the contacts dataframe
        merged_df = contact_df_final.merge(self.get_sets(), on='AGENT', how='inner')
        return merged_df
       
    def get_dials(self):
        """
        Extracts information about dials from the dials dataframe.
        
        Returns:
            A DataFrame with columns for the agent name and their number of dials along with other calculated values.
        """
        
        # Drop the column that represents the agent group, since we don't need it
        self.dials_df.drop(columns={'AGENT GROUP'}, inplace=True)
        
        # Extract the agent username from the email address
        self.dials_df['AGENT'] = self.dials_df['AGENT'].apply(lambda x: x.split("@")[0])

        # Sum the columns that represent dials for each agent
        self.dials_df['Dials'] = self.dials_df.sum(axis=1)

        # Select only the columns we need
        self.dials_df = self.dials_df[['AGENT','AGENT FIRST NAME','AGENT LAST NAME','Dials']]
        
        # Merge the sets dataframe and the dials dataframe using the 'AGENT' column
        merged_df = self.get_contacts().merge(self.dials_df, on='AGENT', how='inner')

        # Concatenate the first and last names to create a full name column
        merged_df['AGENT FULL NAME'] = merged_df['AGENT FIRST NAME'] + ' ' + merged_df['AGENT LAST NAME'].str[0]

        # Calculate the percentage of sets per dials
        merged_df['Sets/Dials'] = round((merged_df['Sets'] / merged_df['Dials']) * 100, 2)

        # Calculate the sets per contact
        merged_df['Sets/Contacts'] = round((merged_df['Sets'] / merged_df['Contacts']), 2)
        
        return merged_df
        
    def calculate_five9_calling_hours(self):
        """
        Calculates the total time each agent spent in a "Ready" state, an "On Call" state, and on the phone ("Five9 calling hours")
        based on the input data frame of Five9 call center agent data.

        Returns:
            A DataFrame with columns for the agent name and their total Five9 calling hours.
        """
        
        # Split the time columns into hours, minutes, and seconds
        self.five9_call_hours[['on_call_hour', 'on_call_min', 'on_call_sec']] = self.five9_call_hours['On Call / AGENT STATE TIME'].str.split(":", expand=True).astype(float)
        self.five9_call_hours[['ready_hour', 'ready_min', 'ready_sec']] = self.five9_call_hours['Ready / AGENT STATE TIME'].str.split(":", expand=True).astype(float)

        # Group the data by agent and calculate total time in fractional hours
        agent_hour_min_sec = self.five9_call_hours.groupby("AGENT").sum()
        agent_hour_min_sec['on_call_min'] = agent_hour_min_sec['on_call_min'].apply(lambda x: x/60)
        agent_hour_min_sec['on_call_sec'] = agent_hour_min_sec['on_call_sec'].apply(lambda x: x/60/60)
        agent_hour_min_sec['ready_min'] = agent_hour_min_sec['ready_min'].apply(lambda x: x/60)
        agent_hour_min_sec['ready_sec'] = agent_hour_min_sec['ready_sec'].apply(lambda x: x/60/60)
        agent_hour_min_sec['total_on_call_time'] = round(agent_hour_min_sec['on_call_hour'] + agent_hour_min_sec['on_call_min'] + agent_hour_min_sec['on_call_sec'], 2)
        agent_hour_min_sec['total_ready_time'] = round(agent_hour_min_sec['ready_hour'] + agent_hour_min_sec['ready_min'] + agent_hour_min_sec['ready_sec'], 2)

        # Calculate the total Five9 calling hours and format the output
        agent_hour_min_sec['total_five9_hours'] = agent_hour_min_sec['total_on_call_time'] + agent_hour_min_sec['total_ready_time']
        agent_hour_min_sec_df = agent_hour_min_sec.reset_index()
        agent_hour_min_sec_df['AGENT'] = agent_hour_min_sec_df['AGENT'].apply(lambda x: x.split("@")[0])
        total_agent_hours = agent_hour_min_sec_df[['AGENT', 'total_five9_hours']]

        return total_agent_hours
        
    def merge_hours_and_dials(self):
        """
        Merge dataframes containing dial data and Five9 calling hour data for each agent.

        Returns:
            Returns a new dataframe containing columns for agent first name, number of dials, 
            number of contacts, number of sets, sets per dial, sets per contact, Five9 calling hours, 
            rounded calling hours, and sets per Five9 calling hours.
        """        
        # Merge previous dataframes
        merged_df = self.get_dials().merge(self.calculate_five9_calling_hours(), on='AGENT', how='inner')
        
        # Select and round columns
        export_df = merged_df[['AGENT FIRST NAME', 'Dials', 'Contacts', 'Sets', 'Sets/Dial', 'Sets/Contact', 'Five9 Calling Hours']]
        export_df = export_df.round({'Sets/Dial': 2, 'Sets/Contact': 2, 'Five9 Calling Hours': 2})
        
        # Compute and round new calling hours column
        export_df['calling_hours_rounded'] = export_df['Five9 Calling Hours'].apply(lambda x: round(x * 4) / 4)
        export_df = export_df.drop(columns=['Five9 Calling Hours'])
        export_df = export_df.rename(columns={'calling_hours_rounded': 'Five9 Calling Hours'})
        
        # Compute and add Sets/Five9 Calling Hours column
        export_df['Sets/Five9 Calling Hours'] = export_df['Sets'] / export_df['Five9 Calling Hours']
        
        return export_df

    def extract_paylocity_working_hours(self):
        """
        Extracts the working hours for each agent from a Paylocity data frame and returns a new data frame with the agents' names
        and their corresponding working hours.

        Returns:
            A DataFrame with columns for the agent name (first name and last initial) and their Paylocity working hours.
        """
        
        # Gather a list of rows containing NaN values
        is_nan = self.payloc_df.isnull()
        rows_with_nan = self.payloc_df[is_nan.any(axis=1)]
        list_of_nan_values = rows_with_nan.iloc[:,4].to_list()

        # Find working hours in the rows with NaN values
        working_hours = []
        for value in list_of_nan_values:
            try:
                if float(value):
                    working_hours.append(value)
            
            # Disregard non-floats
            except ValueError:
                pass

        # Select only the working hours from the list
        working_hours = [x for x in working_hours if pd.isnull(x) == False]

        # Find agents in the rows
        agents = self.payloc_df[self.payloc_df.iloc[:,0].apply(lambda x: "ID:" in x)].iloc[:,1].to_list()

        # Create a DataFrame of found agents and working hours
        df = pd.DataFrame({'Agents':agents, 'Paylocity Working Hours':working_hours})

        # Extract first name and last initial from the agents' names
        df['first_name'] = df['Agents'].apply(lambda x:x.split(',')[1])
        df['first_name'] = df['first_name'].apply(lambda x:x.lower())
        df['first_name'] = df['first_name'].apply(lambda x:x[1].capitalize()+x[2:])
        df['last_initial'] = df['Agents'].apply(lambda x:x[0])

        # Combine first name and last initial into a single column
        df['AGENT FIRST NAME'] = df['first_name'] + ' ' + df['last_initial']

        # Convert the columns to strings
        df[['AGENT FIRST NAME', 'Paylocity Working Hours']] = df[['AGENT FIRST NAME', 'Paylocity Working Hours']].astype(str)

        # Extract the relevant columns and return the resulting DataFrame
        payloc_df_merge = df[['AGENT FIRST NAME', 'Paylocity Working Hours']]
        return payloc_df_merge

    def clean_and_export(self):
        """
        Cleans and merges data from the Five9 and Paylocity dataframes, performs various calculations, and exports the resulting data to an Excel file.

        Returns:
            A DataFrame containing the cleaned and merged data, including various calculated values and additional rows for the mean and total values.
        """
        # Merge dataframes
        merged_df = self.merge_hours_and_dials().merge(self.extract_paylocity_working_hours(), on='AGENT FIRST NAME', how='outer')

        # Calculate Five9 Calling Hours/Paylocity Hours
        merged_df['Five9 Calling Hours/Paylocity Hours'] = (merged_df['Five9 Calling Hours'] / merged_df['Paylocity Working Hours'].astype(float))*100

        # Sort values by Sets column in descending order
        merged_df.sort_values(by='Sets', ascending=False, inplace=True)

        # Calculate means for each column and insert rows into dataframe
        mean_row = ['Avg./Rep', 
                    np.mean(merged_df['Dials']), 
                    np.mean(merged_df['Contacts']), 
                    np.mean(merged_df['Sets']), 
                    np.mean(merged_df['Sets/Dial']), 
                    np.mean(merged_df['Sets/Contact']),
                    np.mean(merged_df['Five9 Calling Hours']), 
                    np.mean(merged_df['Sets/Five9 Calling Hours']),
                    np.mean(merged_df['Paylocity Working Hours'].astype(float)),
                    np.mean(merged_df['Five9 Calling Hours/Paylocity Hours'].astype(float))]

        total_row = ['Total', 
                     np.sum(merged_df['Dials'].iloc[:len(merged_df)-1]), 
                     np.sum(merged_df['Contacts'].iloc[:len(merged_df)-1]), 
                     np.sum(merged_df['Sets'].iloc[:len(merged_df)-1]), 
                     np.mean(merged_df['Sets/Dial'].iloc[:len(merged_df)-1]), 
                     np.mean(merged_df['Sets/Contact'].iloc[:len(merged_df)-1]), 
                     np.sum(merged_df['Five9 Calling Hours'].iloc[:len(merged_df)-1]),
                     np.mean(merged_df['Sets/Five9 Calling Hours'].iloc[:len(merged_df)-1]),
                     np.sum(merged_df['Paylocity Working Hours'].astype(float).iloc[:len(merged_df)-1]),
                     np.mean(merged_df['Five9 Calling Hours/Paylocity Hours'].astype(float).iloc[:len(merged_df)-1])]

        merged_df.loc[-1] = mean_row
        merged_df.loc[-2] = total_row

        # Replace NaN values with empty strings
        merged_df.fillna('', inplace=True)

        # Convert percentage values to strings and round decimal values
        merged_df['Five9 Calling Hours/Paylocity Hours'] = merged_df['Five9 Calling Hours/Paylocity Hours'].apply(lambda x: str(round(x))+"%" if isinstance(x,float) else x)
        merged_df = merged_df.round({'Sets/Dial': 2,'Sets/Contact': 2, 'Five9 Calling Hours': 2, 'Sets/Five9 Calling Hours': 2})

        # Fill NaNs with blanks
        excluded_df.fillna("", inplace=True)

        # Export data to Excel file
        filename = f'weekly_metrics_{self.start_date}_{self.end_date}.xlsx'
        excluded_df.to_excel(filename, index=False)
        return excluded_df
    
# These functions validate inputted dates
def _valid_start_date():
    flag = False
    while flag != True:
        start_date = str(input("Enter the start date: "))
        try:
            pd.to_datetime(start_date)
            return start_date
            break
        except:
            print(f"{start_date} is not a valid date, use format 'yyyy-mm-dd'")

def _valid_end_date():
    flag = False
    while flag != True:
        end_date = str(input("Enter the end date: "))
        try:
            pd.to_datetime(end_date)
            return end_date
            break
        except:
            print(f"{end_date} is not a valid date, use format 'yyyy-mm-dd'")
    
        
# instantiate class
instance = weekly_metrics(call_center_csv="data/call_center_master_list.csv",
                      dials_csv="data/total_warm_dials.csv",
                      contacts_csv="data/total_warm_contacts.csv",
                      five9_csv="data/agent_daily_summary.csv",
                      paylocity_csv="data/master_timecard_summary.csv",
                      start_date = _valid_start_date(),
                      end_date=_valid_end_date())

# run final method
instance.clean_and_export()
