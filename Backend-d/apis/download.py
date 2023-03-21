import os
from django.http import HttpResponse, Http404
from pathlib import Path
from rest_framework.views import APIView

BASE_DIR = Path(__file__).resolve().parent.parent


class download(APIView):
    def get(request, path):
        file_path = os.path.join(BASE_DIR, 'apis/TestValues/factsheet.json')

        if os.path.exists(file_path):
            with open(file_path, 'rb') as fh:
                response = HttpResponse(
                    fh.read(), content_type="application/vnd.ms-excel")
                response['Content-Disposition'] = 'inline; filename=' + \
                    os.path.basename(file_path)
                return response
        raise Http404
