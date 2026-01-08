############## 
import pandas as pd 
import numpy as np
import logging
import os
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


data_folder = os.getenv("DATA_DIR" , "data_folder")
os.makedirs(data_folder, exist_ok=True)

class DataImportation:
    def __init__(self):
        self.df = None 

    def datareading(self):
        file_path = r'src/apartments_for_rent_classified_100K.csv'

        try:
            if os.path.exists(file_path):
                print(f"The path '{file_path}' exists.")
            else:
                print(f"The path '{file_path}' does not exist.")
                
            #file_path = f'Hotel_Reviews.csv'
            self.df = pd.read_csv(file_path , encoding='windows-1252' , sep = ';' )

        except Exception as e:
            logging.error(f"Error while combining files: {e}")
            raise

        return self.df

    def transformation(self):

        try:
            
            self.df['pets_allowed'] = self.df['pets_allowed'].fillna('no_pets_allowed')
            self.df['bedrooms'] = self.df['bedrooms'].fillna(0)
            self.df['datetime_utc'] = pd.to_datetime(self.df['time'], unit='s', utc=True)
            self.df['log_price'] = np.log1p(self.df['price'])
            

            ##### This is working on amenities to change it , I need to write it better
            self.df['amenities_clean'] = self.df['amenities'].fillna('No Amenities')
            self.df['amenities_clean'] = self.df['amenities_clean'].str.split(',')
            self.df['amenities_clean'] = self.df['amenities_clean'].apply(
                    lambda x: [a.strip().lower().replace(' ', '_') for a in x]
                )
            
            mlb = MultiLabelBinarizer()
            amenities_encoded = mlb.fit_transform(self.df['amenities_clean'])

            amenity_freq = amenities_encoded.mean(axis=0)

            keep = (amenity_freq > 0.02) & (amenity_freq < 0.90)
            X_filtered = amenities_encoded[:, keep]
            n_components1 = min(20, X_filtered.shape[1] - 1)
            svd = TruncatedSVD(n_components= n_components1, random_state=42) 
            #### you can format this and see if there is an improvement in the way clusters are made
            X_reduced = svd.fit_transform(X_filtered)
            kmeans = KMeans(n_clusters=5, random_state=42)
            self.df['amenity_cluster'] = kmeans.fit_predict(X_reduced)

            cluster_labels = {
                    0: "Higher_Amenities",
                    1: "Basic_Parking",
                    2: "Low_Amenity",
                    3: "Lifestyle_Recreation",
                    4: "Family_Oriented"
                }

            self.df['amenity_segment'] = self.df['amenity_cluster'].map(cluster_labels)


            ##### let me drop columns that are having a nas since that are fewer 
            ### check the EDA , Latititude and longtitude 25 each and price has 1 
            ### although we need to check these , we can create a regional category out of this
            ### and then seee if it improves our model

            self.df = self.df.dropna()

        except Exception as e:
            logging.error(f"Error while combining files: {e}")
            raise

        return self.df
    

    def remove_columns(self):
        try:
            #### more analysis can be done on these columns but I are removing them 
            ### you can check the EDA work book and find out why I did ths 
            columns_to_remove = [ 
                'source' , 'price_type','amenities_clean','amenity_cluster','category' ,'body',
                'amenities','time','title', 'currency' ,'price_display',
                 'id' ,'cityname' ,'state','datetime_utc'  , 'price',
                 'address'
            ]

            self.df = self.df.drop(          
                columns_to_remove , axis = 1)

            print(self.df.columns)

            rent_cleaned_data_path = os.path.join(data_folder ,"Training_data.csv" )    
            self.df.to_csv(rent_cleaned_data_path)
        
        except Exception as e:
            logging.error(f"Error while combining files: {e}")
            raise

        return self.df


#if __name__ == "__main__":
    #importer = DataImportation()
    #### Function to read data 
    #df = importer.datareading()
    #df = importer.transformation()
    #df=  importer.remove_columns()



