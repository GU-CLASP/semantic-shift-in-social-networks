{ lib
, buildPythonPackage
, fetchPypi
, six
}:
buildPythonPackage rec {
  pname = "jsonlines";
  version = "1.2.0";
  src = fetchPypi {
    inherit pname version;
    sha256 = "126yxzxq7yqhkd3gxmksxrd5kv10iviq15d4lk464j4xi9cdbf23";
  };
  propagatedBuildInputs = [ six ];
  checkInputs = [  ];
  doCheck = false;
}
