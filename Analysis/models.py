import time
from django.db import models
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from django.core import serializers   
from typing import Optional, List

class PandasModelMixin(models.Model):
    class Meta:
        abstract = True

    @classmethod
    def as_dataframe(cls, queryset=None, field_list=None):
        t1 = time.time()

        if queryset is None:
            queryset = cls.objects.all()
        if field_list is None:
            field_list = [_field.name for _field in cls._meta._get_fields(reverse=False)]

        data = []
        [data.append([obj.serializable_value(column) for column in field_list]) for obj in queryset]

        columns = field_list

        df = pd.DataFrame(data, columns=columns)
        print("Execution time without serialization: %s" % time.time()-t1)
        return df

    @classmethod
    def as_dataframe_using_django_serializer(cls, queryset=None):
        t1 = time.time()

        if queryset is None:
            queryset = cls.objects.all()

        if queryset.exists():
            serialized_models = serializers.serialize(format='python', queryset=queryset)
            serialized_objects = [s['fields'] for s in serialized_models]
            data = [x.values() for x in serialized_objects]

            columns = serialized_objects[0].keys()

            df = pd.DataFrame(data, columns=columns)
        df = pd.DataFrame()
        print("Execution time using Django serializer: %s" % time.time()-t1)
        return df

    @classmethod
    def as_dataframe_using_drf_serializer(cls, queryset=None, drf_serializer=None, field_list=None):
        from rest_framework import serializers
        t1 = time.time()

        if queryset is None:
            queryset = cls.objects.all()

        if drf_serializer is None:
            class CustomModelSerializer(serializers.ModelSerializer):
                class Meta:
                    model = cls
                    fields = field_list or '__all__'

            drf_serializer = CustomModelSerializer

        serialized_objects = drf_serializer(queryset, many=True).data
        data = [x.values() for x in serialized_objects]

        columns = drf_serializer().get_fields().keys()

        df = pd.DataFrame(data, columns=columns)
        print("Execution time using DjangoRestFramework serializer: %s" % time.time()-t1)
        return df

class Localized(models.Model):
    title: str
    description: str

    def __init__(self, title: str, description: str) -> None:
        self.title = title
        self.description = description


class Default:
    url: str
    width: int
    height: int

    def __init__(self, url: str, width: int, height: int) -> None:
        self.url = url
        self.width = width
        self.height = height


class Thumbnails(models.Model):
    default: Default
    medium: Default
    high: Default
    standard: Default
    maxres: Default

    def __init__(self, default: Default, medium: Default, high: Default, standard: Default, maxres: Default) -> None:
        self.default = default
        self.medium = medium
        self.high = high
        self.standard = standard
        self.maxres = maxres


class Snippet(models.Model):
    published_at: datetime
    channel_id: str
    title: str
    description: str
    thumbnails: Thumbnails
    channel_title: str
    tags: Optional[List[str]]
    category_id: int
    live_broadcast_content: str
    localized: Localized
    default_language: Optional[str]
    default_audio_language: Optional[str]

    def __init__(self, published_at: datetime, channel_id: str, title: str, description: str, thumbnails: Thumbnails, channel_title: str, tags: Optional[List[str]], category_id: int, live_broadcast_content: str, localized: Localized, default_language: Optional[str], default_audio_language: Optional[str]) -> None:
        self.published_at = published_at
        self.channel_id = channel_id
        self.title = title
        self.description = description
        self.thumbnails = thumbnails
        self.channel_title = channel_title
        self.tags = tags
        self.category_id = category_id
        self.live_broadcast_content = live_broadcast_content
        self.localized = localized
        self.default_language = default_language
        self.default_audio_language = default_audio_language


class Statistics(models.Model):
    view_count: int
    like_count: int
    favorite_count: int
    comment_count: int

    def __init__(self, view_count: int, like_count: int, favorite_count: int, comment_count: int) -> None:
        self.view_count = view_count
        self.like_count = like_count
        self.favorite_count = favorite_count
        self.comment_count = comment_count


class Item(models.Model):
    kind: str
    etag: str
    id: str
    snippet: Snippet
    statistics: Statistics

    def __init__(self, kind: str, etag: str, id: str, snippet: Snippet, statistics: Statistics) -> None:
        self.kind = kind
        self.etag = etag
        self.id = id
        self.snippet = snippet
        self.statistics = statistics


class PageInfo(models.Model):
    total_results: int
    results_per_page: int

    def __init__(self, total_results: int, results_per_page: int) -> None:
        self.total_results = total_results
        self.results_per_page = results_per_page


class Welcome2(PandasModelMixin):
    kind: str
    etag: str
    items: List[Item]
    next_page_token: str
    page_info: PageInfo

    def __init__(self, kind: str, etag: str, items: List[Item], next_page_token: str, page_info: PageInfo) -> None:
        self.kind = kind
        self.etag = etag
        self.items = items
        self.next_page_token = next_page_token
        self.page_info = page_info

class Welcome2s(models.Model):
    Welcome: Welcome2

    def __init__(self, Welcome: Welcome2) -> None:
        self.Welcome = Welcome