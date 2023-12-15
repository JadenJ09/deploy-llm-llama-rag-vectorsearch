
# %%
import pkgutil

def list_submodules(module):
    return [modname for importer, modname, ispkg in pkgutil.iter_modules(module.__path__)]

# Example usage:
import azure.storage.filedatalake as adl
print(list_submodules(adl))

# %%

# %%
"""
1. **Data Ingestion (Azure Data Lake)**:
   - Store and organize PDFs in Azure Data Lake.
   - Set up automated ingestion pipelines using Azure Data Factory or similar services.
   
# Create a new resource group to hold the storage account -
# if using an existing resource group, skip this step
az group create --name my-resource-group --location westus2

# Install the extension 'Storage-Preview'
az extension add --name storage-preview

# Create the storage account
az storage account create --name my-storage-account-name --resource-group my-resource-group --sku Standard_LRS --kind StorageV2 --hierarchical-namespace true

# Create the file system
az storage fs create --account-name <my-storage-account-name> --name <my-file-system-name> --account-key <my-storage-account-key>

# Check the file system
az storage fs list --account-name <my-storage-account-name> --account-key <my-storage-account-key> --query "[].name"

# Create creation with a connection string
from azure.storage.filedatalake import DataLakeServiceClient
service = DataLakeServiceClient.from_connection_string(conn_str="my_connection_string")

"""
from azure.storage.filedatalake import DataLakeServiceClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# az account list-locations --query "[].{Region:name, DisplayName:displayName}" -o table

# Set up your Azure Data Lake account credentials
account_name = os.getenv("account_name")
account_key = os.getenv("account_key")
file_system_name = os.getenv("file_system_name")

# Create client
service = DataLakeServiceClient(account_url=f"https://{account_name}.dfs.core.windows.net/", credential=account_key)


# Create a DataLakeFileSystemClient object
file_system_client = service_client.get_file_system_client(file_system_name)

# Specify the local path of the file
local_file_path = os.getenv("local_file_path")

# Specify the destination path in Azure Data Lake
destination_path = os.getenv("destination_path")

# Upload the file to Azure Data Lake
with open(local_file_path, "rb") as file:
    file_client = file_system_client.get_file_client(destination_path)
    file_client.upload_data(file.read(), overwrite=True)

# Download the file from Azure Data Lake
downloaded_file = file_client.download_file().readall()

# Write the downloaded data to a file
with open("downloaded_file.txt", "wb") as file:
    file.write(downloaded_file)

# %%
"""
2. **Data Conversion and Processing (PySpark)**:
   - Convert PDFs to text using Python libraries. (e.g. 'PyPDF2')
   - Leverage PySpark for data cleaning and preparation.
"""
import PyPDF2
from pyspark.sql import SparkSession

def convert_pdf_to_text(file_path):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    text = ''
    for page_num in range(pdf_reader.numPages):
        page_obj = pdf_reader.getPage(page_num)
        text += page_obj.extractText()
    pdf_file_obj.close()
    return text

# Initialize a SparkSession
spark = SparkSession.builder.getOrCreate()

# Assuming df is your DataFrame
df = spark.read.text("downloaded_file.txt")

# Drop rows with null values
df = df.na.drop()

# Show the DataFrame
df.show()
