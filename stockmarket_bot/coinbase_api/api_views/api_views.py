import requests
from django.http import JsonResponse
import json
import datetime
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view

from coinbase_api.models import Cryptocurrency
from coinbase_api.serializers import NaNJSONEncoder,CryptocurrencySerializer, BitcoinSerializer, EthereumSerializer, PolkadotSerializer
from ..cb_auth import Granularities
from rest_framework import viewsets, filters
from rest_framework.renderers import JSONRenderer
from rest_framework.views import APIView
from django_filters import rest_framework as django_filters
from ..models import AbstractOHLCV, Bitcoin, Ethereum, Polkadot
from ..utilities.utils import cb_fetch_product_list, cb_fetch_product_candles
from ..views.views import cb_fetch_product



@api_view(['GET'])
def cb_fetch_product_list_view(request):
    """
    @param limit - A limit describing how many products to return.\n
    @param offset - Number of products to offset before returning.\n
    @param product_type (ENUM: SPOT or FUTURE) - Type of products to return.\n
    @param product_ids - List of product IDs to return.\n
    @param contract_expiry_type - ENUM: UNKNOWN_CONTRACT_EXPIRY_TYPE or EXPIRING\n
    Get a list of the available currency pairs for trading.\n
    datastructure: \n
    products\n
    --[id]\n
    ----product_id\n
    ----price\n
    ----price_percentage_change_24h\n
    ----volume_24h\n
    ----volume_percentage_change_24h\n
    ----base_increment\n
    ----quote_increment\n
    ----quote_min_size\n
    ----quote_max_size\n
    ----base_min_size\n
    ----base_max_size\n
    ----base_name\n
    ----quote_name\n
    ----watched\n
    ... for more see result\n
    num_products
    """
    try:
        data = cb_fetch_product_list()
        try:
            tmp = json.loads(data.content)
            products = tmp['products']
            db_data = Cryptocurrency.objects.all()
            db_data.delete()
            changed_products = []
            for item in products:
                item['quote_display_symbol'] = item['product_id'].split('-')[1]
                item['base_display_symbol'] = item['product_id'].split('-')[0]
                changed_products.append(item)
            serializer = CryptocurrencySerializer(data=changed_products, many=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except KeyError:
            print('KeyError!')
            pass
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse(data)


@api_view(['GET'])
def cb_fetch_product_view(request):
    query_params = request.query_params
    product_id = query_params.get('product_id')
    if product_id == None:
        return Response({'Error':f'key product_id not found inside query params: {query_params}'}, status=500)
    data = cb_fetch_product(product_id)
    tmp = json.loads(data.content)
    return Response(tmp, status=status.HTTP_200_OK)




@api_view(['GET'])
def cb_fetch_product_candles_view(request):
    query_params = request.query_params
    print(query_params)
    product_id = query_params.get('product_id') 
    print(f'product_id: {product_id}')
    start = query_params.get('start') 
    end = query_params.get('end')
    granularity = query_params.get('granularity')
    if granularity is None:
        granularity = Granularities.ONE_HOUR.value
    if end is None and start is None:
        end = int(datetime.datetime.timestamp(datetime.datetime.now()))
        start = int(datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(hours=300)))
    if start is None and end is not None:
        start = end - datetime.timedelta(hours=300)
    if start is not None and end is None:
        end = start + datetime.timedelta(hours=300)
    if product_id is None:
        product_id = 'BTC-EUR'
    data = cb_fetch_product_candles(product_id, start, end, granularity)
    tmp = json.loads(data.content)
    return Response(tmp, status=status.HTTP_200_OK)

class OHLCVFilter(django_filters.FilterSet):
    timestamp = django_filters.DateTimeFromToRangeFilter()
    open = django_filters.RangeFilter()
    high = django_filters.RangeFilter()
    low = django_filters.RangeFilter()
    close = django_filters.RangeFilter()
    volume = django_filters.RangeFilter()

    class Meta:
        model = AbstractOHLCV
        fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

class NaNJSONRenderer(JSONRenderer):
    encoder_class = NaNJSONEncoder

class AbstractOHLCVView(viewsets.ReadOnlyModelViewSet):  # assuming you only want to read
    renderer_classes = (NaNJSONRenderer, )
    filter_backends = (django_filters.DjangoFilterBackend, filters.OrderingFilter)
    filterset_class = OHLCVFilter
    ordering_fields = '__all__'

class BitcoinView(AbstractOHLCVView):
    renderer_classes = (NaNJSONRenderer, )
    queryset = Bitcoin.objects.all()
    serializer_class = BitcoinSerializer

class EthereumView(AbstractOHLCVView):
    queryset = Ethereum.objects.all()
    serializer_class = EthereumSerializer

class PolkadotView(AbstractOHLCVView):
    queryset = Polkadot.objects.all()
    serializer_class = PolkadotSerializer
