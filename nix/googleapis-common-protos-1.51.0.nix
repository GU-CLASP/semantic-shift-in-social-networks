{ lib
, buildPythonPackage
, fetchPypi
, protobuf
}:
buildPythonPackage rec {
  pname = "googleapis-common-protos";
  version = "1.51.0";
  src = fetchPypi {
    inherit pname version;
    sha256 = "0vi2kr0daivx2q1692lp3y61bfnvdw471xsfwi8924br89q92g01";
  };
  propagatedBuildInputs = [ protobuf ];
  checkInputs = [  ];
  doCheck = false;
}
