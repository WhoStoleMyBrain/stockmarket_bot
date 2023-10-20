import requests
from django.http import JsonResponse
import json
import datetime
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from coinbase_api.serializers import AccountSerializer, CryptocurrencySerializer, BitcoinSerializer, EthereumSerializer, PolkadotSerializer, PredictionSerializer, BitcoinViewSerializer
from ..cb_auth import Granularities
from rest_framework import viewsets, filters
# from rest_framework.renderers import JSONRenderer
# from rest_framework.views import APIView
from django_filters import rest_framework as django_filters
from ..models.models import Cryptocurrency, AbstractOHLCV, Account, Bitcoin, Ethereum, Polkadot, Prediction
from ..utilities.utils import cb_fetch_product_list, cb_fetch_product_candles
from ..views.views import cb_fetch_product
from rest_framework.views import APIView
import datetime
from django.utils import timezone



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

# class NaNJSONRenderer(JSONRenderer):
#     encoder_class = NaNJSONEncoder

class AbstractOHLCVView(viewsets.ReadOnlyModelViewSet):  # assuming you only want to read
    # renderer_classes = (NaNJSONRenderer, )
    filter_backends = (django_filters.DjangoFilterBackend, filters.OrderingFilter)
    filterset_class = OHLCVFilter
    ordering_fields = '__all__'

class BitcoinView(AbstractOHLCVView):
    # renderer_classes = (NaNJSONRenderer, )
    queryset = Bitcoin.objects.all()
    serializer_class = BitcoinSerializer

class EthereumView(AbstractOHLCVView):
    queryset = Ethereum.objects.all()
    serializer_class = EthereumSerializer

class PolkadotView(AbstractOHLCVView):
    queryset = Polkadot.objects.all()
    serializer_class = PolkadotSerializer

class PredictionFilter(django_filters.FilterSet):
    timestamp_predicted_for = django_filters.DateTimeFromToRangeFilter()
    timestamp_predicted_at = django_filters.DateTimeFromToRangeFilter()
    model_name = django_filters.RangeFilter()
    predicted_field = django_filters.RangeFilter()
    crypto = django_filters.RangeFilter()
    predicted_value = django_filters.RangeFilter()

    class Meta:
        model = Prediction
        fields = ['timestamp_predicted_for', 'timestamp_predicted_at', 'model_name', 'predicted_field', 'crypto', 'predicted_value']

class PredictionView(viewsets.ReadOnlyModelViewSet):  # assuming you only want to read
    # renderer_classes = (NaNJSONRenderer, )
    filter_backends = (django_filters.DjangoFilterBackend, filters.OrderingFilter)
    filterset_class = PredictionFilter
    ordering_fields = '__all__'
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer


class BitcoinData(APIView):
    def get(self, request):
        one_week_ago = timezone.now() - datetime.timedelta(days=7)
        data = Bitcoin.objects.filter(timestamp__gte=one_week_ago).order_by('timestamp')
        serializer = BitcoinViewSerializer(data, many=True)
        return Response(serializer.data)
    
class AccountViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Account.objects.all()
    serializer_class = AccountSerializer


class CryptocurrencyFilter(django_filters.FilterSet):
    quote_display_symbol = django_filters.CharFilter()
    base_display_symbol = django_filters.CharFilter()

    class Meta:
        model = Cryptocurrency
        fields = ['quote_display_symbol', 'base_display_symbol']


class CryptocurrencyViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Cryptocurrency.objects.all()
    serializer_class = CryptocurrencySerializer
    filterset_class = CryptocurrencyFilter
    filter_backends = (django_filters.DjangoFilterBackend, filters.OrderingFilter)
    ordering_fields = '__all__'

    # @action(detail=False, methods=['get'])
    # def fetch_product_list(self, request, *args, **kwargs):
    #     try:
    #         data = cb_fetch_product_list()
    #         try:
    #             tmp = json.loads(data.content)
    #             products = tmp['products']
    #             self.queryset.delete()
    #             changed_products = []
    #             for item in products:
    #                 item['quote_display_symbol'] = item['product_id'].split('-')[1]
    #                 item['base_display_symbol'] = item['product_id'].split('-')[0]
    #                 changed_products.append(item)
    #             serializer = self.get_serializer(data=changed_products, many=True)
    #             if serializer.is_valid():
    #                 serializer.save()
    #                 return Response(serializer.data, status=status.HTTP_201_CREATED)
    #             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    #         except KeyError:
    #             print('KeyError!')
    #             pass
    #     except requests.RequestException as e:
    #         return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    #     return Response(data)