{ lib
, buildPythonPackage
, fetchPypi
, google-auth
, googleapis-common-protos
, requests
, grpc
, grpcio
}:
buildPythonPackage rec {
  pname = "google-api-core";
  version = "1.16.0";
  extra = "grpc";
  src = fetchPypi {
    inherit pname version;
    sha256 = "1qh30ji399gngv2j1czzvi3h0mgx3lfdx2n8qp8vii7ihyh65scj";
  };
  propagatedBuildInputs = [ google-auth googleapis-common-protos requests grpc grpcio ];
  checkInputs = [  ];
  doCheck = false;
}
