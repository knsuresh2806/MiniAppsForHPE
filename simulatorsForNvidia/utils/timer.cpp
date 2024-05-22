
#include <cstdio>
#include "timer.h"
#include <map>
#include <algorithm>

typedef std::map<std::string, timer*> MapType;

static MapType _registry;

static bool timerSort( timer *t1, timer *t2 ) {
   return t1->get_elapsed() > t2->get_elapsed();
}

timer & timer::get_timer( std::string name ) {
    MapType::iterator i = _registry.find(name);

    if ( _registry.end() == i ) {
        timer *t = new timer(name);
       _registry[name] = t;
       return *t;
    }

    return *(i->second);
}

std::vector<timer *> timer :: get_timers() {
  std::vector<timer *> timers;

  for (MapType::iterator i = _registry.begin(); i != _registry.end(); ++i)
    timers.push_back(new timer(i->second));

  return timers;
}

void timer :: flush_timers() {
    for (MapType::iterator i = _registry.begin(); i!=_registry.end(); ++i)
       delete i->second;

    _registry.clear();
}

void timer::print_timers( FILE *stream, timer &total ) {
    std::vector<timer*> timers = get_timers();

    std::sort( timers.begin(), timers.end(), timerSort );

    for ( size_t i=0; i<timers.size(); i++ ) {
        timer *t = timers[i];
        fprintf(stream, "%9.2fs/%5.1f%% %9.2fs/%5.1f%% %9.2fs/%5.1f%%  [%s]\n",
              t->get_elapsed(), 100*t->get_elapsed()/total.get_elapsed(),
              t->get_user(),    100*t->get_user()/total.get_user(),
              t->get_system(),  100*t->get_system()/total.get_system(),
              t->get_name().c_str() );
    }

    for ( size_t i=0; i<timers.size(); i++ )
        delete timers[i];
}


