# -*- coding: utf-8 -*-
# Stuart Johnstone 07/2020
# Opal auction historic data scraper - looks at all the stores on the site and
# gets the opal data from the reviews left - this includes that actual price sold for
import os
import requests
from bs4 import BeautifulSoup as bs
import bs4
import pandas as pd
from re import sub
from decimal import Decimal
from matplotlib import pyplot as plt
import webbrowser
import shutil
import time

import datasets
import argparse

data_storage = datasets.data_storage
# select true if you only want to look at the no reserve auction. Select false if you want to scrape data for learning
startPage = 1
pages = 2 # Number of pages to search per store
stores = 2


# start by getting a list of store pages to look at.
starting_url="https://www.opalauctions.com/sitemap"
site_map = requests.get(starting_url)
site_map = bs(site_map.text, "lxml")
url_list = []
store_initials = "5ABCDEFGHIKLMNOPQRSTUVWY"
# each initial has a row id - in that there are several store links
for c in store_initials:
    # if there is a type of opal with that initial then the store will be the second element, if not it'll be the first
    try:
        initial = site_map.find_all(id=c)[1]
    except IndexError:
        initial = site_map.find_all(id=c)[0]
    for link in initial.find_all('a'):
        url_list.append(link.get('href'))

print("There are {} stores to search".format(len(url_list)))

opals = []
sale_urls = []
for opal_store_url in url_list[:stores]:
    for i in range(pages):
        if i == 0:
            appendix = "#feedback"
        else:
            appendix = "/all/"+str(i)
        # navigate to a listing
        opal_store_feedback_url=opal_store_url.replace("/stores/","/profile/",1)+appendix
        print("\n"+opal_store_feedback_url)
        listing = requests.get(opal_store_feedback_url)
        listing = bs(listing.text, "lxml")
        # print(listing)
        feedback_divs = listing.find_all("div", "panel panel-sm panel-feedback panel-default items")
        # print(feedback_divs)
        for div in feedback_divs:
            # print(div)
            link = div.find_all("a")[1].get("href")
            # print(link)
            sale_urls.append(link)

print("There are {} opals to collect data from".format(len(sale_urls)))
for sold in sale_urls[:]:
    o = datasets.Opal()
    o.extract_detail_data(sold)
    o.deets()
    opals.append(o)

print("\n******* finshed extracting data********\n")

df = datasets.create_dataframe(opals)
df.to_csv(datasets.pred_file)
print("New ", df.shape[0])
# print(df[["ID", "Title", "Timestamp"]])

#open old file, append and delete old version of duplicated entries
try:
    # if old version does not exist, just write a new file
    old_df = pd.read_csv(data_storage, index_col=0)
    print("Old ", old_df.shape[0])
    # print(old_df[["ID", "Title", "Timestamp"]])
    df = pd.concat([old_df, df], sort=False, ignore_index=True)
    print("Concat ",df.shape[0])
    # print(df[["ID", "Title", "Timestamp"]])
    df = df.sort_values(["ID", "Timestamp"])
    print("sorted ",df.shape[0])
    # print(df[["ID", "Title", "Timestamp"]])
    df.drop_duplicates(["ID", "Title"], keep='last', inplace=True)
    print("Droppped ", df.shape[0])
    print(df[["ID", "Title", "Timestamp"]])
except:
    print('error reading old file')
    pass



df.to_csv(data_storage)
print("The training file is {} long".format(df.shape[0]))

#plot data
# df.hist(column="Weight (carats)", bins=20)
# plt.show()



def plot_points(df, colour = False):
    '''
    :param df: data frame of opals
    :param colour: string to use as colour map for plot
    :return:
    '''
    fig,ax = plt.subplots()
    marker = "." # "o" ","
    ax.set_xlabel = "Weight (carats)"
    ax.set_ylabel = "Price (USD)"
    if colour:
        # Drop all record where Body tone isn't available
        df=df[df["Body Tone"] != "NA"]
        df = df[df["Weight (carats)"] < 100]
        # create a dict that gives the body tone string an integer number
        bt_dict = {}
        for i in range(8):
            bt_dict[str(i) + " N"] = i
        # assign color to a new colum by mapping body tone using dict
        df["color"] = df["Body Tone"].map(bt_dict)
        print(df)

        sc=plt.scatter(df["Weight (carats)"], df["Price (USD)"], marker=marker, c=df["color"], cmap=plt.cm.get_cmap(colour))
        cb=fig.colorbar(sc, ax=ax)
        cb.set_label("Body tone *N ({})".format(colour))
        ax.set_facecolor('xkcd:black')
    else:
        sc=plt.scatter(df["Weight (carats)"], df["Price (USD)"], marker=marker)
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    urls = df['URl'].tolist()
    names = df['Title'].tolist()

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                text = "{}".format(" ".join([urls[n] for n in ind["ind"]]))
                webbrowser.open(text, new=2)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()


    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

# plot_points(df)
cols = ['viridis', 'plasma', 'inferno', 'magma', 'cividis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cols = ['plasma']
for c in cols[:]:
    plot_points(df, colour=c)

