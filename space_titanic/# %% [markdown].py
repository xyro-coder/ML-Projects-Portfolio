# %% [markdown]
# ### Importing the data

# %%
import os
import tarfile
import urllib
#declare the link we are downloading from
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
#Create a local path where the house dataset will be stored
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_URL):
    #using os we create a for the housing csv if it exsits we don't create a copy
    os.makedirs(housing_path,exist_ok=True)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    #this using urlib to download data from housing url and save it locally at tgz_path
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# %%
fetch_housing_data()

# %% [markdown]
# ### since the data was in tgz form we must first open it with gzip then load it into a pandas dataframe

# %%
import pandas as pd
import gzip

def load_housing_data():
    csv_path = "/home/icaarus/Desktop/deep learning/datasets/housing/housing.csv.gz"
    with gzip.open(csv_path, 'rt', encoding='utf-8') as file:
        return pd.read_csv(file)

housing = load_housing_data()







# %%
housing.head()

# %%
housing.info()

# %%
housing['ocean_proximity'].value_counts()

# %%
#this function is used to infer and get a feel for the data 
housing.describe()

# %%

import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()

# %% [markdown]
# ### lets create a test set from the data

# %%
import numpy as np
def split_train_test(data,test_ratio):
    # this creates a new array of random permutations of length data which are used as the indices for the data
    shuffled_indices = np.random.permutation(len(data))
    # we cut the data by the test_ratio and assign it to the variable test set size 
    test_set_size = int(len(data) * test_ratio)
    #the test indices are equal to the first index of shuffled array to the test-size number index of shuffled array
    test_indices = shuffled_indices[:test_set_size]
    #the train indices are equal to from test-size number index of shuffled arry to the last index
    train_indices = shuffled_indices[test_set_size:]
     # Return the training and testing data using the calculated indices
    return data.iloc[train_indices], data.iloc[test_indices]

# %% [markdown]
# lets now spilt the data into train and test 

# %%
train_set, test_set = split_train_test(housing, 0.2)

# %%
len(train_set)

# %%
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda x: test_set_check(x, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# %%
train_set

# %%
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# %%
train_set

# %%
housing["income_cat"] = pd.cut(housing['median_income'],bins= [0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1,2,3,4,5])
housing['income_cat'].hist()

# %%
from sklearn.model_selection import StratifiedShuffleSplit

# Handle missing values in 'income_cat' column
housing.dropna(subset=['income_cat'], inplace=True)

# Stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]



# %%
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# %%
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# %% [markdown]
# # Insights :lets explore our data

# %%
housing = strat_train_set.copy()

# %%
housing.rename(columns={'housing.csv': 'longitude'}, inplace=True)
housing['longitude']

# %%
housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Use a diverging color map for better contrast
cmap = plt.get_cmap('jet')

# Adjusting the size of each point based on the median income
size = housing['population'] / 100

# Creating a scatter plot
scatter_plot = plt.scatter(
    x=housing['longitude'],
    y=housing['latitude'],
    alpha=0.4,
    s=size,
    c=housing["median_house_value"],
    cmap=cmap,
    label='population'
)

# Adding a color bar legend
cbar = plt.colorbar(scatter_plot, label='Median House Value', aspect=40, pad=0.1)

# Adding a gradient legend for population size
legend_size = plt.Line2D([0], [0], marker='o', color='w', label='Population', markersize=10, linestyle='None', markerfacecolor='black', alpha=0.6)
plt.legend(handles=[legend_size], loc='upper right', fontsize=12)

# Adding a title and labels
plt.title('Housing Prices and Population in California', fontsize=18)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)

# Customizing the grid appearance
plt.grid(linestyle='dashed', linewidth=0.5, alpha=0.5)

# Adding a background color for better contrast
plt.gca().set_facecolor('#f0f0f0')

# Displaying the plot
plt.show()


# %%
corr_matrix = housing.corr()

# %%
corr_matrix["median_house_value"].sort_values(ascending=False)

# %%
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes],figsize=(10,7))

# %%
housing.plot(kind="scatter", x="median_income", y="median_house_value",s=housing['median_income'],c=housing['bedrooms_per_room'],cmap=plt.get_cmap('jet'),alpha=1,colorbar=True)

# %%
corr_matrix.plot()

# %%
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

# %%
housing = strat_train_set.drop('median_house_value',axis=1)
housing_label = strat_train_set['median_house_value'].copy()

# %%
housing_label

# %% [markdown]
# # Data cleaning

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
housing_num = housing.drop('ocean_proximity',axis=1)
imputer.fit(housing_num)
housing.fillna(housing_num)


# %%
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)
housing_tr

# %%
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

# %%
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]

# %%
ordinal_encoder.categories_

# %%
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot.toarray()

# %%
from sklearn.base import BaseEstimator, TransformerMixin
room_ix, bedroom_ix, population_ix, household_ix = 3,4,5,6
class CombinedAttributeAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedroom_per_room=True):
        self.add_bedroom_per_room = add_bedroom_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        room_per_household = X[:,room_ix] / X[:,household_ix]
        population_per_household = X[:,population_ix] / X[:,household_ix]
        if self.add_bedroom_per_room:
            bedroom_per_room = X[:,bedroom_ix] / X[:,room_ix]
            return np.c_[X,room_per_household,population_per_household,bedroom_per_room]
        else:
            return np.c_[X,room_per_household,population_per_household]
attr_adder = CombinedAttributeAdder(add_bedroom_per_room=False)
housing_extra_attributes = attr_adder.transform(housing.values)
housing_extra_attributes

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributeAdder()),
        ('std_scaler', StandardScaler()),
                        ])
housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr

# %%
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)

# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_label)


# %%
housing_label

# %%


# %%


# %%
some_data = housing.iloc[:5]
some_labels = housing_label.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Prediction",lin_reg.predict(some_data_prepared))


# %%
print("Labels:", list(some_labels))

# %%
from sklearn.metrics import mean_squared_error
housing_prediction = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_label, housing_prediction)
lim_rmse = np.sqrt(lin_mse)
lim_rmse

# %%
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_label)

# %%
housing_prediction = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_label,housing_prediction)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# %%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg,housing_prepared,housing_label,scoring='neg_mean_squared_error',cv= 10)
tree_rmse_scores = np.sqrt(-scores)


# %%
display_scores(tree_rmse_scores)

# %%
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_label)

# %%
forest_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_label,forest_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

# %%
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg,param_grid,cv= 5,scoring='neg_mean_squared_error',return_train_score= True)
grid_search.fit(housing_prepared,housing_label)

# %%
grid_predict = grid_search.predict(housing_prepared)
grid_mse = mean_squared_error(housing_label,grid_predict)
grid_rmse = np.sqrt(grid_mse)
grid_rmse

# %%
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

# %%
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# %%
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,730.2


# %%
from sklearn.svm import SVR
vec_reg = SVR(kernel= 'linear', C=10000)
vec_reg.fit(housing_prepared,housing_label)


# %%
def rmse(model):
    return np.sqrt(mean_squared_error(housing_label,model.predict(housing_prepared)))


# %%
rmse(vec_reg)

# %%
from sklearn.model_selection import RandomizedSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_searchi = RandomizedSearchCV(forest_reg,param_grid,cv= 5,scoring='neg_mean_squared_error',return_train_score= True)
grid_searchi.fit(housing_prepared,housing_label)

# %%
rmse(grid_searchi)

# %%
rmse(grid_search)

# %%



