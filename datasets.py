# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
import requests
from bs4 import BeautifulSoup as bs
from re import sub
from decimal import Decimal
from matplotlib import pyplot as plt
import webbrowser
import shutil
import time
import pickle
import imutils
import math

os.chdir('C:/Users/stuar/PycharmProjects/face_det/venv/Opals_v2')
data_storage = "opal_data3.csv"
pred_file = "predictions.csv"

maxPrice = 1000
maxWeight = 50
pic_size = 64
lines = -1


def time_to_sec(tl):
     time_s = int(tl[:tl.find("d")]) * 24 * 36000 + int(tl[tl.find("d ")+2:tl.find("h")]) * 3600 +int(tl[tl.find("h ")+2:tl.find("m")]) * 60 +int(tl[tl.find("m ")+2:tl.find("s")])

class Opal:

    def __init__(self):
        self.title = ""
        self.url = ""
        self.ID = ""
        self.weight = 0 # carats
        self.price = 0
        self.shipping_cost = None # worst case cost in case shipping in not listed
        self.tax = 0
        self.seller = ""
        self.type = ""
        self.image_url = ""
        self.local_img_fn = ""
        self.dims = [0, 0, 0]
        self.body_tone = ""
        self.auction_time = 0
        self.time = time.time()
        self.sold_date = 0

    def extract_data(self, link):
        ol=str(link)
        #print(ol)
        heading = self.extract_between(ol, "<h3", "</h3>")
        heading = self.extract_between(heading, "<a", ">")
        self.url = (self.extract_between(heading, 'href="', '"') + "/").rstrip()
        self.title = self.extract_between(heading, 'title="', '"').replace(',','').replace('/','-')

        # get id from
        ad_id = self.extract_between(ol, '"end-bid" style="display: none;">', '</span>')
        self.ID = self.extract_between(ad_id, 'data-id="', '"') 

        # try to get price
        self.price = self.extract_between(ol, '-shopping-cart"></i>', '</a>').strip()
        if ol.find('<span class="end-bid"') >= 0:
            self.price = self.extract_between(ol, '"end-bid" style="display: none;">', '</span>')
            self.auction_time = self.extract_between(ol, 'seconds-remaining="', '"')

        # try to get image
        image_url = self.extract_between(ol, 'thumbnail-image">', "</a>")
        self.image_url = self.extract_between(image_url, 'src="', '"/>')
        try:
            self.download_img()
        except:
            self.local_img = "Error"

        # now go to that page and extract new data
        opal_page = requests.get(self.url)
        opal_page = bs(opal_page.text, "lxml")

        self.get_data_table(opal_page)
        self.get_shipping_tax(opal_page)

    def extract_detail_data(self, link):
        # this gets all the data from the actual site
        self.url = link
        opal_page = requests.get(link)
        opal_page = bs(opal_page.text, "lxml")
        self.get_data_table(opal_page)
        self.get_shipping_tax(opal_page)
        self.get_title(opal_page)
        self.get_id(opal_page)
        self.type = self.get_type(opal_page)
        self.get_price(opal_page)
        self.get_image(opal_page)
        self.get_sold_on_date(opal_page)

    def get_title(self, opal_page):
        self.title = opal_page.find_all("h1", "panel-title")[0].get_text().strip()
        return self.title

    def get_id(self, opal_page):
        auid=opal_page.find("div", "pull-right plain text-gray hidden-xs").get_text()
        self.ID = auid[-6:]
        return self.ID

    def get_type(self, opal_page):
        breadcrumb = opal_page.find("ol", "breadcrumb").get_text()
        # print(breadcrumb)
        types = {
        	"black": "Black Opal",
        	"boulder": "Boulder Opal",
        	"coober": "Coober Pedy Opal",
        	"crystal": "Crystal Opal",
        	"dark": "Dark Opal Semi Black",
        	"ethiopian": "Ethiopian Opal",
        	"mexican": "Mexican Fire Opal",
        }
        for type in types:
        	if types[type] in breadcrumb:
        		return type
        return False

    def get_price(self, opal_page):
        if opal_page.find("h3", "text-danger"):
            bids = opal_page.find("div", "bids")
            price = bids.find("td").get_text()
            self.price = price
            self.auction_time = 0
        elif opal_page.find("a", "btn btn-success btn-lg btn-block btn-toggle-cart btn-add-to-cart"):
            price =  opal_page.find("div", "panel panel-info panel-item-primary").find("h3", "panel-title")
            self.price = price[-4:]
            self.auction_time = 0
        elif opal_page.find("span", "countdown"):
            tl = opal_page.find("span", "countdown").find("span").get_text()
            time_s = time_to_sec(tl)
            self.auction_time = time_s
            bids = opal_page.find("div", "bids")
            price = bids.find("td").get_text()
            self.price = price
        else:
            self.price = "$0"
        return self.price

    def get_image(self, opal_image):
        self.image_url = opal_image.find("a", "gallery-pic").get("href")
        try:
        	self.download_img()
        except:
        	self.local_img_fn = "Error"

    # TODO: add sold on date
    def get_sold_on_date(self, opal_page):
        #print(opal_page)
        try:
            self.sold_date = opal_page.find("td", "ends text-right").text
        except:
            pass

    def get_data_table(self, opal_page):
        data_table = str(opal_page.find_all('table')[0])
        #get weight
        try:
           weight = self.extract_between(data_table, 'Weight', "</tr>")
           weight = self.extract_between(weight, '<td>', "carats")
           self.weight = float(weight)
        except:
           pass
        #get dimensions
        try:
            dim = self.extract_between(data_table, 'Dimensions', "</tr>")
            dim = self.extract_between(dim, '<td>', "mm")
            self.dims = [float(i) for i in dim.split("x")]
        except:
           pass
        #get body tone
        try:
            bt = self.extract_between(data_table, 'Body Tone</strong>', "</tr>")
            self.body_tone = self.extract_between(bt, '<td>', "</td>")
            if "</strong>" in self.body_tone:
                self.body_tone = "NA"
        except:
           pass

    def get_shipping_tax(self, opal_page):
        # get shipping costs
        # TODO: Fix shipping
        try:
            shipping_table = str(opal_page.find_all("div", 'shipping')[0])
            #print("Shipping table", shipping_table)
            self.shipping_cost = self.currency_float(self.extract_between(shipping_table, 'Shipping', "<br/>").strip())
        except IndexError:
            list_tables = opal_page.find_all("table", 'table table-condensed')
            # from all the tables found in the above class - find the shipping one
            for table in list_tables:
                if table.find("th"):
                    #print("===========> tables ", table)
                    headings = [t.text for t in table.find_all('th')]
                    # print("Headings", headings)
                    if "Shipping" in headings:
                        rates = [t.text.strip() for t in table.find_all('td', 'sel-shipping-rate text-right')]
                        # print("rates", rates)
                        self.shipping_cost =  min(map(self.currency_float,rates))



        # get tax costs
        tax_table = opal_page.find_all("div",'panel-body')
        for panel_body in tax_table:
            if "tax of " in str(panel_body):
               self.tax = float(self.extract_between(str(panel_body), 'tax of ', "%."))/100
               break


       
    def deets(self):
         print("ID: {}".format(self.ID))
         print(self.url)
         print(self.title)
         print("Type: {}".format(self.type))
         print(self.price)
         print("Shipping cost: {}".format(self.shipping_cost))
         print("Tax rate: {:d}%".format(int(self.tax*100)))
         print("Stone weighs {} carats".format(self.weight))
         print("Stone dimensions are {} mm".format(self.dims))
         print("Body tone is {}".format(self.body_tone))
         print("Time left on auction is {} s".format(self.auction_time))
         print("Image url: {}".format(self.image_url))
         print("Sold on: {}".format(self.sold_date))
         print("\n")


    def extract_between(self, t, a, b):
        t_start = t.find(a)
        #print(t_start)
        t = t[t_start + len(a):]
        t_end = t.find(b)
        t = t[:t_end]
        return t.rstrip().strip()

    def price_float(self):
        value = Decimal(sub(r'[^\d.]', '', self.price))
        return float(value)

    @staticmethod
    def currency_float(currency):
        print("Currency in cur float", currency)
        return float(currency[1:])

    def download_img(self):
        folder="opal_imgs/"
        resp = requests.get(self.image_url, stream=True)
        # Open a local file with wb ( write binary ) permission.
        fn  = folder + self.ID + "_" + self.title[:20] + '.png'
        self.local_img_fn = fn
        local_file = open(fn, 'wb')
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        resp.raw.decode_content = True
        # Copy the response stream raw data to local image file.
        shutil.copyfileobj(resp.raw, local_file)
        # Remove the image url response object.
        del resp


###Functions####

def create_dataframe(opals):
	# Create dataframe from a list of Opal ojects
	data_list = []
	for o in opals:
		data_list.append([o.ID, o.price_float(), o.shipping_cost, o.tax, o.type, o.weight, o.dims[0], o.dims[1], o.dims[2], o.body_tone, o.url, o.title, o.local_img_fn, o.time, o.auction_time, o.sold_date])

	df = pd.DataFrame(data_list, columns=["ID", "Price (USD)", "Shipping Cost", "Tax rate", "Type", "Weight (carats)", "Length", "Width", "Height", "Body Tone", "URl", "Title", "Image Filename", "Timestamp", "Auction Time", "Sold Date"])
	print(df)
	return df


###Model functions###
def load_opal_attributes(inputPath, load=True, augment_multi=0):
	# load data from csv using Pandas
	df = pd.read_csv(inputPath, index_col=0)
	df=df.iloc[:lines]
	#count unique body tones
	bdy_tn = df["Body Tone"].value_counts().keys().tolist()
	print("The body tone keys in dataset are {}".format(bdy_tn))
	counts = df["Body Tone"].value_counts().tolist()
	df.reset_index(inplace=True)
	df = df.drop("index", axis=1)
	
	#Get rid of instance with images that cannot be read 
	print("Size before dropping dodgy images {}".format(df.shape))
	to_drop = []
	for i, imPath in enumerate(df["Image Filename"].tolist()):
		im = cv2.imread(imPath)
		if im is None:
			#print("{} is none!!!".format(imPath))
			to_drop.append(i)
		elif im.size==0:
			print("Size0")
			to_drop.append(i)	
	df = df.drop(to_drop, axis=0)

	print("Size after dropping dodgy images {}".format(df.shape))
	idxs = df[df['Weight (carats)'] > maxWeight].index
	df.drop(idxs, inplace=True)
	idxs = df[df['Price (USD)'] > maxPrice].index
	df.drop(idxs, inplace=True)
	print("Size after dropping opals outside max val {}".format(df.shape))
	
	if load:
		for (zipcode, count) in zip(bdy_tn, counts):
		# If there are certain body tones that only have a few instances, remove them
			if count < 10:
				idxs = df[df['Body Tone'] == zipcode].index
				df.drop(idxs, inplace=True)

		bdy_tn = df["Body Tone"].value_counts().keys().tolist()
		print("The body tone keys in dataset are {} after sanitisation".format(bdy_tn))
		# drop lines where the items are still on auction or only have a few seconds left, as these will be much cheaper than they woud
		idxs = df[df['Auction Time'] > 5].index
		df.drop(idxs, inplace=True)

		# pickle list so that predictor can use it
		with open('body_tone.pickle', 'wb') as f:
			pickle.dump(bdy_tn, f, pickle.HIGHEST_PROTOCOL)
		print("Pickled")
	
	print("Size AFTER dropping dodgy images {}".format(df.shape))
	# return the data frame
	# repeat lines a certain number of time
	if augment_multi >= 1:
		df = balance_data(df, 6)
	if augment_multi > 1.5:
		print("Augmenting data....")
		aug_df = pd.DataFrame(np.repeat(df.values, augment_multi, axis=0))
		aug_df.columns = df.columns
		df = aug_df
	return df

def balance_data(df, bins):
	print("Balacing data ...")
	#df = df.iloc[:200]
	hist_out = df.hist(column=["Price (USD)"], bins=bins)
	#plt.show()
	sizes = []
	binned_dfs = []
    # TODO: if max is >> than rest remove some of max
	for bin in range(bins):
		bin_min = bin*maxPrice/bins
		bin_max = (bin+1)*maxPrice/bins
		df_bin = df[(df["Price (USD)"] >= bin_min) & (df["Price (USD)"] < bin_max)]
		df_length =  df_bin.shape
		sizes.append(df_length[0])
		binned_dfs.append(df_bin)
		print("There are {} opals in price bin {} to {} AUD".format(df_length, bin_min, bin_max))

	max_bin = max(sizes)
	print(max_bin)

	for i, size in enumerate(sizes):
		if size ==0:
			size=1
		if i == 0:
			print("zero")
			balanced_df = binned_dfs[i]
		else:
			if size == max_bin:
				print("max")
				balanced_df = concat([balanced_df, binned_dfs[i]], sort=False, ignore_index=True)
			else:
				print("else")
				mul = int(math.floor(max_bin/size))
				print("mul", mul)
				aug = pd.DataFrame(np.repeat(binned_dfs[i].values, mul, axis=0))
				aug.columns = balanced_df.columns
				balanced_df = pd.concat([balanced_df, aug], sort=False, ignore_index=True)
				print(balanced_df.columns)
		print("Size of balanced data: {}".format(balanced_df.shape))
	print(balanced_df)
	a = balanced_df[(balanced_df["Price (USD)"]<4)]
	print(a)
	balanced_df[["Price (USD)"]]=balanced_df[["Price (USD)"]].apply(pd.to_numeric)
	hist_out = balanced_df.hist(column=["Price (USD)"], bins=bins)
	#plt.show()
	return balanced_df



def process_opal_attributes(df, train, test):
	# initialize the column names of the continuous data
	continuous1 = ["Weight (carats)"]
	continuous2 = ["Length", "Width", "Height"]

	max_dim = 40
	trainContinuous1 = train[continuous1] / maxWeight
	testContinuous1 = test[continuous1] / maxWeight

	trainContinuous2 = train[continuous2] / max_dim
	testContinuous2 = test[continuous2] / max_dim

	# one-hot encode the zip code categorical data (by definition of
	# one-hot encoding, all output features are now in the range [0, 1])
	with open('body_tone.pickle', 'rb') as f:
		body_tone_list = pickle.load(f)
	print("Pickle loaded")

	zipBinarizer = LabelBinarizer().fit(body_tone_list)
	trainCategorical = zipBinarizer.transform(train["Body Tone"])
	testCategorical = zipBinarizer.transform(test["Body Tone"])
	# construct our training and testing data points by concatenating
	# the categorical features with the continuous features
	trainX = np.hstack([trainCategorical, trainContinuous1, trainContinuous2])
	testX = np.hstack([testCategorical, testContinuous1, testContinuous2])
	# return the concatenated training and testing data
	print("Training data")
	print(trainX)
	return (trainX, testX)

def zoom(img, lim_max):
	h, w, d = img.shape
	#print("Image height: {}, Image width: {}".format(h,w))
	scale = np.random.rand()*(lim_max-1)+1
	#print("Scale {}".format(scale))
	h = int((h - h/scale)/2+1)
	w = int((w - w/scale)/2+1)
	img = img[h:-h, w:-w]
	return img

def load_opal_images(imgdf, augment_multi=1):
	# initialize our images array (i.e., the house images themselves)
	images = []

	img_list = imgdf["Image Filename"].tolist()

	print(imgdf.shape)
	for img in img_list:
		#print(img)

		image = cv2.imread(img)
		if augment_multi > 1:
			image = zoom(image, 1.8)

		try:
			image = cv2.resize(image, (pic_size, pic_size))
		except:
			print("faulty image: {}".format(img))
		if augment_multi > 1:
			r = np.random.random_integers(4) * 90
			f = np.random.random_integers(3)-1		
			#print("Rotation angle = {}".format(r))
			image = imutils.rotate(image, angle=r)
			if f <2:
				image = cv2.flip(image, f)
			 		
		#cv2.imshow("im", image)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows() 
		images.append(image)
	return np.array(images)

