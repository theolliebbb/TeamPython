from django.shortcuts import render
from django.http import HttpResponse
import requests
import urllib.request
from django.views import generic
from . import models
import json
import numpy as np
import pandas as pd
import random
from bokeh.models import ColumnDataSource
from urllib.request import urlopen
from bokeh.plotting import figure
from bokeh.plotting import figure, show
from bokeh.embed import components
from bokeh.models import LinearAxis, Range1d
from bokeh.transform import dodge

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=20,10

from keras.layers import LSTM,Dropout,Dense
url = 'https://youtube.googleapis.com/youtube/v3/videos?part=id&part=snippet&part=statistics&chart=mostPopular&maxResults=100000&key=AIzaSyDaHhgnxRUkv4kFUaS8mqJmr3WI3ijyF4c'
url2 = 'https://youtube.googleapis.com/youtube/v3/videos?part=id&part=snippet&part=statistics&chart=mostPopular&maxResults=100000&pageToken=CDIQAA&key=AIzaSyDaHhgnxRUkv4kFUaS8mqJmr3WI3ijyF4c'
response = urlopen(url)
requestjson = json.loads(response.read())
response2 = urlopen(url2)
requestjson2 = json.loads(response2.read())

class IndexView(generic.ListView):
    template_name = 'index.html'
    def get_queryset(self):
        return requestjson

class ml(generic.ListView):
    template_name = 'ml.html'
    def get_queryset(self):
        return requestjson


def mllink(request):
    return ml.html
def graphresults(request):
    requests = request.GET.get("q")
    requestsCat = request.GET.get("p")
    requestsSort = request.GET.get("r")
    requestsCoun = request.GET.get("s")
    try:
        if requestsCat == "" and requestsCoun == "":
            url = 'https://youtube.googleapis.com/youtube/v3/videos?part=id&part=snippet&part=statistics&chart=mostPopular&maxResults=100000&key=AIzaSyDaHhgnxRUkv4kFUaS8mqJmr3WI3ijyF4c'
            url2 = 'https://youtube.googleapis.com/youtube/v3/videos?part=id&part=snippet&part=statistics&chart=mostPopular&maxResults=100000&pageToken=CDIQAA&key=AIzaSyDaHhgnxRUkv4kFUaS8mqJmr3WI3ijyF4c'
        elif requestsCat == "" and requestsCoun != "":
            url = 'https://youtube.googleapis.com/youtube/v3/videos?part=id&part=snippet&part=statistics&chart=mostPopular&regionCode={}&maxResults=100000&key=AIzaSyDaHhgnxRUkv4kFUaS8mqJmr3WI3ijyF4c'.format(requestsCoun)
            url2 = 'https://youtube.googleapis.com/youtube/v3/videos?part=id&part=snippet&part=statistics&chart=mostPopular&regionCode={}&maxResults=100000&pageToken=CDIQAA&key=AIzaSyDaHhgnxRUkv4kFUaS8mqJmr3WI3ijyF4c'.format(requestsCoun)
        elif requestsCoun == "":
            url = 'https://youtube.googleapis.com/youtube/v3/videos?part=id&part=snippet&part=statistics&chart=mostPopular&maxResults=100000&videoCategoryId={}&key=AIzaSyDaHhgnxRUkv4kFUaS8mqJmr3WI3ijyF4c'.format(requestsCat)
            url2 = 'https://youtube.googleapis.com/youtube/v3/videos?part=id&part=snippet&part=statistics&chart=mostPopular&maxResults=100000&pageToken=CDIQAA&videoCategoryId={}&key=AIzaSyDaHhgnxRUkv4kFUaS8mqJmr3WI3ijyF4c'.format(requestsCat)
        else:
            url = 'https://youtube.googleapis.com/youtube/v3/videos?part=id&part=snippet&part=statistics&chart=mostPopular&regionCode={}&maxResults=100000&videoCategoryId={}&key=AIzaSyDaHhgnxRUkv4kFUaS8mqJmr3WI3ijyF4c'.format(requestsCoun, requestsCat)
            url2 = 'https://youtube.googleapis.com/youtube/v3/videos?part=id&part=snippet&part=statistics&chart=mostPopular&regionCode={}&maxResults=100000&pageToken=CDIQAA&videoCategoryId={}&key=AIzaSyDaHhgnxRUkv4kFUaS8mqJmr3WI3ijyF4c'.format(requestsCoun, requestsCat)
        response = urlopen(url)
        requestjson = json.loads(response.read())
        response2 = urlopen(url2)
        requestjson2 = json.loads(response2.read())
        vid_list = []
        vid_list = getstatscondition(requestjson2, getstatscondition(requestjson, vid_list, requests), requests)
        titles=[]
        counts = []
        likes = []
        statisticslist = []
        snippetlist = []
        if requestsSort == "Views":
            vid_list.sort(key=lambda x: int(x.statistics['viewCount']))
            vid_list.reverse()
        elif requestsSort == "Likes":
            for o in vid_list:
                try:
                    x = o.statistics['likeCount']
                except:
                    o.statistics['likeCount'] = 0;
            vid_list.sort(key=lambda x: int(x.statistics['likeCount']))
            vid_list.reverse()
        else:
            vid_list = vid_list
        comments, titles, counts, likes, links = sortresults(vid_list)
            
        modifiedCounts = []
        for o in counts:
            modifiedCounts.append(int(o) / 1000)
        modifiedLikes = []
        for o in likes:
            modifiedLikes.append(int(o) /1000)
        widthamount = 0
        if len(titles) == 1:
            widthamount = 300
        else:
            widthamount = len(titles) * 150

        data = {
            'titles': titles,
            'view': modifiedCounts,
            'like': modifiedLikes}
        source = ColumnDataSource(data = data)
        plot = figure(x_range=titles, height=900, width=widthamount, y_axis_label = 'Views (/1000)', title="Likes And Views", toolbar_location="left",  sizing_mode='scale_height')
        plot.vbar(x=dodge('titles', 0.0, range=plot.x_range), top= 'view', width=0.7, source=source, legend_label="Views")
        plot.extra_y_ranges['like'] = Range1d(start=0, end=max(modifiedLikes))
        plot.add_layout(LinearAxis(y_range_name='like', axis_label='Likes (/1000)'), 'left')
        plot.vbar(x=dodge('titles', 0.25, range=plot.x_range), top='like', width=0.7, source=source, color="#e84d60", legend_label="Likes")
        plot.xaxis.major_label_orientation = .5
        plot.x_range.range_padding = 0.03
        plot.xgrid.grid_line_color = None
        plot.legend.location = "top_left"
        plot.legend.orientation = "horizontal"
        script, div = components(plot)
        return render(request, 'base.html', {'script': script, 'div': div})
    except:
        html = '<html><body style="background-color: #377eda"><center><font color="white">No results with search term {} in category {}.</font></body></html>' .format(requests, requestsCat)
        return HttpResponse(html)
        
# def comment(request):
#     requestPhrase = request.GET.get("o")
#     from .machine import machinelearning
#     requestPhrase2 = machinelearning.generate_text(requestPhrase)
#     requestPhrase3 = machinelearning.generate_text(requestPhrase)
#     requestPhrase4 = machinelearning.generate_text(requestPhrase)
#     requestPhrase5 = machinelearning.generate_text(requestPhrase)
#     requestPhrase6 = machinelearning.generate_text(requestPhrase)
#     requestPhrase7 = machinelearning.generate_text(requestPhrase)
#     try:
#         html = '<html><body style="background-color: #377eda"><center><font color="white">Newly generated comments: {}<br>{}<br>{}<br>{}<br>{}<br>{}.</font></body></html>' .format(requestPhrase2, requestPhrase3, requestPhrase4, requestPhrase5, requestPhrase6, requestPhrase7)
#         return HttpResponse(html)
#     except:
#         html = '<html><body style="background-color: #377eda"><center><font color="white">Error occurred.</font></body></html>' 
#         return HttpResponse(html)

# def description(request):
#     requestPhrase = request.GET.get("a")
#     from .machine import machinelearningDesc
#     requestPhrase2 = machinelearningDesc.generate_text(requestPhrase)
#     requestPhrase3 = machinelearningDesc.generate_text(requestPhrase)
#     requestPhrase4 = machinelearningDesc.generate_text(requestPhrase)
#     requestPhrase5 = machinelearningDesc.generate_text(requestPhrase)
#     requestPhrase6 = machinelearningDesc.generate_text(requestPhrase)
#     requestPhrase7 = machinelearningDesc.generate_text(requestPhrase)
#     try:
#         html = '<html><body style="background-color: #377eda"><center><font color="white">Newly generated comments: {}<br>{}<br>{}<br>{}<br>{}<br>{}.</font></body></html>' .format(requestPhrase2, requestPhrase3, requestPhrase4, requestPhrase5, requestPhrase6, requestPhrase7)
#         return HttpResponse(html)
#     except:
#         html = '<html><body style="background-color: #377eda"><center><font color="white">Error occurred.</font></body></html>' 
#         return HttpResponse(html)

# def title(request):
#     requestPhrase = request.GET.get("b")
#     from .machine import machinelearningbigT
#     requestPhrase2 = machinelearningbigT.generate_text(requestPhrase)
#     requestPhrase3 = machinelearningbigT.generate_text(requestPhrase)
#     requestPhrase4 = machinelearningbigT.generate_text(requestPhrase)
#     requestPhrase5 = machinelearningbigT.generate_text(requestPhrase)
#     requestPhrase6 = machinelearningbigT.generate_text(requestPhrase)
#     requestPhrase7 = machinelearningbigT.generate_text(requestPhrase)
#     try:
#         html = '<html><body style="background-color: #377eda"><center><font color="white">Newly generated comments: {}<br>{}<br>{}<br>{}<br>{}<br>{}.</font></body></html>' .format(requestPhrase2, requestPhrase3, requestPhrase4, requestPhrase5, requestPhrase6, requestPhrase7)
#         return HttpResponse(html)
#     except:
#         html = '<html><body style="background-color: #377eda"><center><font color="white">Error occurred.</font></body></html>' 
#         return HttpResponse(html)

# def videopage(request):
#     requestTitle = request.GET.get("c")
#     requestDesc = request.GET.get("d")
#     requestCom = request.GET.get("e")
    
#     file = r'Analysis\results.json'
#     response = open(file, encoding="utf8")
#     requestjson = json.load(response)
#     vid_list = getstats(requestjson, vid_list=[])
#     comments, titles, counts, likes, links = sortresults(vid_list)
#     requestViews = random.randint(100000, 1000000000)
#     viewresult, commentresult, likeresult = getratios(titles, comments, likes, counts, requestViews)

#     base = "https://www.youtube.com/embed/"
    
#     dislikeresult = likeresult[0]/10
#     from .machine import machinelearningbigT
#     from .machine import machinelearning
#     from .machine import machinelearningDesc
#     title = machinelearningbigT.generate_text(requestTitle)
#     description = machinelearningDesc.generate_text(requestDesc)
#     comment = machinelearning.generate_text(requestCom)
#     driver = webdriver.Chrome('chromedriver.exe')
#     baseurl = "http://youtube.com"
#     driver.get(f'{baseurl}/search?q={title}')
#     hrefs = [video.get_attribute('href') for video in driver.find_elements(By.ID,"thumbnail")]
#     list = []
#     for href in hrefs:
#         if href is not None:
#             if 'https://www.youtube.com/watch?v=' in href:
#                 newhref = href.replace('https://www.youtube.com/watch?v=', "")
#                 list.append(base + newhref)
#     embed = ""
#     try:
#         embed=list[0]
#     except:
#         embed=""
#     driver.quit()

#     return render(request, 'videopage.html', {'title': title, 'comment': comment, 'description': description, 'likeresult' : likeresult[0], 'requestviews' : viewresult[0], 'commentresult' : commentresult, 'dislikeresult' : dislikeresult, 'id': embed})
    
def data1(request):
    requestViews = request.GET.get("m")
    file = r'Analysis\results.json'
    response = open(file, encoding="utf8")
    requestjson = json.load(response)
    vid_list = getstats(requestjson, vid_list=[])
    comments, titles, counts, likes, links = sortresults(vid_list)
    viewresult, commentresult, likeresult = getratios(titles, comments, likes, counts, requestViews)
    titles = []
    titles.append("Sample Video")
    data = {
            'titles': titles,
            'view': viewresult,
            'like': likeresult,
            'comment': commentresult}
    source = ColumnDataSource(data = data)
    plot = figure(x_range=titles, height=900, width=500, y_axis_label = 'Views / 1000', title="Projected Comments and Like for Selected Views", toolbar_location="left")
    plot.vbar(x=dodge('titles', 0.0, range=plot.x_range), top= 'view', width=0.7, source=source, legend_label="Views")
    plot.extra_y_ranges['like'] = Range1d(start=0, end=max(likeresult))
    plot.extra_y_ranges['comment'] = Range1d(start=0, end=max(commentresult))
    plot.add_layout(LinearAxis(y_range_name='like', axis_label='Projected Likes'), 'left')
    plot.add_layout(LinearAxis(y_range_name='comment', axis_label='Projected Comments'), 'left')
    plot.vbar(x=dodge('titles', 0.12, range=plot.x_range), top='like', width=0.7, source=source, color="#e84d60", legend_label="Likes")
    plot.vbar(x=dodge('titles', 0.25, range=plot.x_range), top='comment', width=0.7, source=source, color="#5e03fc", legend_label="Comments")
    plot.xaxis.major_label_orientation = 3.141/4
    plot.x_range.range_padding = 0.03
    plot.xgrid.grid_line_color = None
    plot.legend.location = "top_left"
    plot.legend.orientation = "horizontal"
    script, div = components(plot)
    return render(request, 'base.html', {'script': script, 'div': div})

def getratios(titles, comments, likes, counts, requestViews):
    viewlikeratio = 0
    viewcommentratio = 0
    invalidcount = 0
    for index in range(0, len(titles)):
        if int(likes[index]) != 0:
            viewlikeratio += float(counts[index]) / float(likes[index])
        else:
            invalidcount = invalidcount + 1
    viewlikeratio = viewlikeratio / (float(len(titles) - invalidcount))
    invalidcount = 0
    for index in range(0, len(titles)):
        if int(comments[index]) != 0:
            viewcommentratio += float(counts[index]) / float(comments[index])
        else:
            invalidcount = invalidcount + 1
    viewcommentratio = viewcommentratio / (float(len(titles) - invalidcount))
    likeresult = [] 
    likeresult.append(int(float(requestViews) / float(viewlikeratio)))
    commentresult = []
    commentresult.append(int(float(requestViews) / float(viewcommentratio)))
    viewresult = []
    viewresult.append(float(requestViews)/1000)

    return viewresult, commentresult, likeresult

def sortresults(vid_list):
    comments = []
    titles=[]
    counts = []
    likes = []
    statisticslist = []
    snippetlist = []
    links = []
    for o in vid_list:
        statisticslist.append(o.statistics)
        snippetlist.append(o.snippet)
        titles.append(o.snippet['title'])
        links.append(o.id)
    for o in statisticslist:
        counts.append(o['viewCount'])
    for o in statisticslist:
        try:
            comments.append(o['commentCount'])
        except:
            comments.append(0)
    for o in statisticslist:
        try:
            likes.append(o['likeCount'])
        except:
            likes.append(0)
    return comments, titles, counts, likes, links

def getstats(jsonfile, vid_list):
    for var in jsonfile['items']:
        wel_item = models.Item(
        kind=var['kind'],
        etag=var['etag'],
        id = var['id'],
        snippet=var['snippet'],
        statistics=var['statistics']
        )
        vid_list.append(wel_item)
    return vid_list

def getstatscondition(jsonfile, vid_list, q):
    for var in jsonfile['items']:
        wel_item = models.Item(
        kind=var['kind'],
        etag=var['etag'],
        id = var['id'],
        snippet=var['snippet'],
        statistics=var['statistics']
        )
        if q.lower() in wel_item.snippet['title'].lower():
                vid_list.append(wel_item)
        else:
            continue
    return vid_list
    

