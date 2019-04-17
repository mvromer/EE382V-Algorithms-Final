#include <iostream>
#include <tclap/CmdLine.h>

int main( int argc, char **argv )
{
	try
	{
		TCLAP::CmdLine cmd( "EE382V Algorithms Final - Matrix Multiply" );

		cmd.parse( argc, argv );
		return 0;
	}
	catch (const TCLAP::ArgException & e)
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}

	return 1;
}
