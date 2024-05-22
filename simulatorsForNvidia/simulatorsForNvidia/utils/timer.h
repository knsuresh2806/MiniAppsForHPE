#ifndef _TIMER_
#define _TIMER_

#include <mpi.h>
#include <sys/times.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <cstring>

#define TIME_TOTAL "Total Elapsed"

#define TIMED_BLOCK(name) \
   for( bool zz_tbc=true; \
        zz_tbc && timer::get_timer(name).start(); \
        zz_tbc = false, timer::get_timer(name).stop() )

#define TIMED_BLOCK_LOCAL(timer) \
   for( bool zz_tbc=true; \
        zz_tbc && timer.start(); \
        zz_tbc = false, timer.stop() )

class timer {

    public:
        timer() : _name("UNNAMED") {
            reset();
        }

        timer(timer *t) {
            _name = t->_name;
            memcpy((void*)&_total, (void*)&(t->_total), sizeof(struct tms));
            memcpy((void*)&_start, (void*)&(t->_start), sizeof(struct tms));
            memcpy((void*)&_stop, (void*)&(t->_stop), sizeof(struct tms));
            _rStart = t->_rStart;
            _rTotal = t->_rTotal;
        }

        timer( std::string name ) : _name(name) {
            reset();
        }

        inline bool start() {
            _rStart = MPI_Wtime();
            return true;
        }

        inline bool stop() {
            _rTotal += MPI_Wtime() - _rStart;
            _total.tms_utime += (_stop.tms_utime - _start.tms_utime);
            _total.tms_stime += (_stop.tms_stime - _start.tms_stime);
            return true;
        }

        inline void reset() {
            memset(&_total, 0, sizeof(struct tms));
            memset(&_start, 0, sizeof(struct tms));
            memset(&_stop, 0, sizeof(struct tms));
            _rTotal = _rStart = 0;
        }

        inline std::string &get_name() {
            return _name;
        }

        inline double get_elapsed() {
            return _rTotal;
        }

        inline double get_user() {
            return _total.tms_utime/(float)sysconf(_SC_CLK_TCK);
        }

        inline double get_system() {
            return _total.tms_stime/(float)sysconf(_SC_CLK_TCK);
        }

        inline void add(timer *timer) {
            _rTotal += timer->_rTotal;
            _total.tms_utime += timer->_total.tms_utime;
            _total.tms_stime += timer->_total.tms_stime;
        }

        inline void print( FILE *file ) {
            fprintf(file, "%9.2f %9.2f %9.2f   [%s]\n",
                get_elapsed(), get_user(), get_system(), _name.c_str() );
        }

        static void flush_timers();

        static void print_timers(FILE *stream, timer &total);

        static timer& get_timer( std::string name );

        //static timer& get_timer( std::string name, int edge );
        //static std::vector<timer *> get_timers(bool aggregate=false);

        static std::vector<timer *> get_timers();

    private:
        std::string _name;
        /// Total time calculated after call to stop()
        struct tms _total;
        /// Start time written after call to start()
        struct tms _start;
        /// Stop time written after call to stop()
        struct tms _stop;

        double _rStart;
        double _rTotal;
};

#endif

